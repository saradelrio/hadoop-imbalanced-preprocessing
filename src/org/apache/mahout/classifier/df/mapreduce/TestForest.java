/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.df.mapreduce;

import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.Arrays;

import com.google.common.io.Closeables;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.RegressionResultAnalyzer;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tool to classify a Dataset using a previously built Decision Forest
 */
public class TestForest extends Configured implements Tool {

  private static final Logger log = LoggerFactory.getLogger(TestForest.class);

  private FileSystem dataFS;
  private Path dataPath; // test data path

  private Path datasetPath;

  private Path modelPath; // path where the forest is stored

  private FileSystem outFS;
  private Path outputPath; // path to predictions file, if null do not output the predictions

  private boolean analyze; // analyze the classification results ?

  private boolean useMapreduce; // use the mapreduce classifier ?
  
  private String dataName;
  
  private long time;

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = DefaultOptionCreator.inputOption().create();

    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true).withArgument(
      abuilder.withName("dataset").withMinimum(1).withMaximum(1).create()).withDescription("Dataset path")
        .create();

    Option modelOpt = obuilder.withLongName("model").withShortName("m").withRequired(true).withArgument(
        abuilder.withName("path").withMinimum(1).withMaximum(1).create()).
        withDescription("Path to the Decision Forest").create();

    Option outputOpt = DefaultOptionCreator.outputOption().create();

    Option analyzeOpt = obuilder.withLongName("analyze").withShortName("a").withRequired(false).create();

    Option mrOpt = obuilder.withLongName("mapreduce").withShortName("mr").withRequired(false).create();

    Option helpOpt = DefaultOptionCreator.helpOption();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(datasetOpt).withOption(modelOpt)
        .withOption(outputOpt).withOption(analyzeOpt).withOption(mrOpt).withOption(helpOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption("help")) {
        CommandLineUtil.printHelp(group);
        return -1;
      }

      dataName = cmdLine.getValue(inputOpt).toString();
      String datasetName = cmdLine.getValue(datasetOpt).toString();
      String modelName = cmdLine.getValue(modelOpt).toString();
      String outputName = cmdLine.hasOption(outputOpt) ? cmdLine.getValue(outputOpt).toString() : null;
      analyze = cmdLine.hasOption(analyzeOpt);
      useMapreduce = cmdLine.hasOption(mrOpt);

      if (log.isDebugEnabled()) {
        log.debug("inout     : {}", dataName);
        log.debug("dataset   : {}", datasetName);
        log.debug("model     : {}", modelName);
        log.debug("output    : {}", outputName);
        log.debug("analyze   : {}", analyze);
        log.debug("mapreduce : {}", useMapreduce);
      }

      dataPath = new Path(dataName);
      datasetPath = new Path(datasetName);
      modelPath = new Path(modelName);
      if (outputName != null) {
        outputPath = new Path(outputName);
      }
    } catch (OptionException e) {
      log.warn(e.toString(), e);
      CommandLineUtil.printHelp(group);
      return -1;
    }
    
    time = System.currentTimeMillis();
    
    testForest();
    
    time = System.currentTimeMillis() - time;
    
    writeToFileClassifyTime(DFUtils.elapsedTime(time));

    return 0;
  }

  private void testForest() throws IOException, ClassNotFoundException, InterruptedException {

    // make sure the output file does not exist
    if (outputPath != null) {
      outFS = outputPath.getFileSystem(getConf());
      if (outFS.exists(outputPath)) {
        throw new IllegalArgumentException("Output path already exists");
      }
    }

    // make sure the decision forest exists
    FileSystem mfs = modelPath.getFileSystem(getConf());
    if (!mfs.exists(modelPath)) {
      throw new IllegalArgumentException("The forest path does not exist");
    }

    // make sure the test data exists
    dataFS = dataPath.getFileSystem(getConf());
    if (!dataFS.exists(dataPath)) {
      throw new IllegalArgumentException("The Test data path does not exist");
    }

    if (useMapreduce) {
      mapreduce();
    } else {
      sequential();
    }

  }

  private void mapreduce() throws ClassNotFoundException, IOException, InterruptedException {
    if (outputPath == null) {
      throw new IllegalArgumentException("You must specify the ouputPath when using the mapreduce implementation");
    }

    Classifier classifier = new Classifier(modelPath, dataPath, datasetPath, outputPath, getConf());

    classifier.run();

    if (analyze) {
      double[][] results = classifier.getResults();
      if (results != null) {
    	//writePredictions(results);
        Dataset dataset = Dataset.load(getConf(), datasetPath);
        if (dataset.isNumerical(dataset.getLabelId())) {
          RegressionResultAnalyzer regressionAnalyzer = new RegressionResultAnalyzer();
          regressionAnalyzer.setInstances(results);
          log.info("{}", regressionAnalyzer);
        } else {
          ResultAnalyzer analyzer = new ResultAnalyzer(Arrays.asList(dataset.labels()), "unknown");
          for (double[] res : results) {
            analyzer.addInstance(dataset.getLabelString(res[0]),
              new ClassifierResult(dataset.getLabelString(res[1]), 1.0));
          }
          log.info("{}", analyzer);
          parseOutput(analyzer);
        }
      }
    }
  }
  
  private void writePredictions(double results[][]) throws IOException {
      NumberFormat decimalFormatter = new DecimalFormat("0.########");
      outFS = outputPath.getFileSystem(getConf());
      FSDataOutputStream ofile = null;
      Path filenamePath = new Path(outputPath, "Predictions").suffix(".txt");
      try   
      {               
        if (ofile == null) {
          // this is the first value, it contains the name of the input file
          ofile = outFS.create(filenamePath);
          // write the Confusion Matrix                                       
          StringBuilder returnString = new StringBuilder();   
         
 
          for (double[] res : results) {
               // returnString.append(res[1]+"\n");
              String dato = Double.toString(res[1])+"\n";
              ofile.writeBytes(dato);

          }
         
          ofile.close();         
        }        
      }
      finally
      {
        Closeables.closeQuietly(ofile);
      }
    } 

  private void parseOutput(ResultAnalyzer analyzer) throws IOException {
    NumberFormat decimalFormatter = new DecimalFormat("0.########");
	outFS = outputPath.getFileSystem(getConf());
	FSDataOutputStream ofile = null;
	int pos=dataName.indexOf('t');
	String subStr=dataName.substring(0, pos);
	Path filenamePath = new Path(outputPath, subStr + "_confusion_matrix").suffix(".txt");
	try    
	{	        	
      if (ofile == null) {
	    // this is the first value, it contains the name of the input file
		ofile = outFS.create(filenamePath);
		// write the Confusion Matrix	      	      	      	      
		StringBuilder returnString = new StringBuilder(200);	      
		returnString.append("=======================================================").append('\n');
		returnString.append("Confusion Matrix\n");
		returnString.append("-------------------------------------------------------").append('\n');
		int [][] matrix = analyzer.getConfusionMatrix().getConfusionMatrix();	      
		for(int i=0; i< matrix.length-1; i++){
		  for(int j=0; j< matrix[i].length-1; j++){	          	          
		    returnString.append(
		                    StringUtils.rightPad(Integer.toString(matrix[i][j]), 5)).append('\t');	
		  } 	        
		  returnString.append('\n');
		}
		returnString.append("-------------------------------------------------------").append('\n');	      	      
		returnString.append("Sensisivity or True Positive Rate (TPR)\n");
		returnString.append(
		    		  StringUtils.rightPad(decimalFormatter.format(computeSensisivityAndSpecificity(matrix)[0]), 5)).append('\n');  
		returnString.append("-------------------------------------------------------").append('\n');	      	      
		returnString.append("Specificity or True Negative Rate (TNR)\n");
		returnString.append(
		    		  StringUtils.rightPad(decimalFormatter.format(computeSensisivityAndSpecificity(matrix)[1]), 5)).append('\n'); 
		returnString.append("-------------------------------------------------------").append('\n');	      	      
		returnString.append("AUC - Area Under the Curve ROC\n");
		returnString.append(
		    		  StringUtils.rightPad(decimalFormatter.format(computeAuc(matrix)), 5)).append('\n');                  
		returnString.append("-------------------------------------------------------").append('\n');	      
		returnString.append("GM - Geometric Mean\n");
		returnString.append(
		    		  StringUtils.rightPad(decimalFormatter.format(computeGM(matrix)), 5)).append('\n');                  
		returnString.append("-------------------------------------------------------").append('\n');
		returnString.append("FM - F-Measure\n");
		returnString.append(
		    		  StringUtils.rightPad(decimalFormatter.format(computeFMeasure(matrix)), 5)).append('\n');                  
		returnString.append("-------------------------------------------------------").append('\n');
		String output = returnString.toString();
		ofile.writeUTF(output);
		ofile.close();		  
	  } 	    
	} 
	finally 
    {
      Closeables.closeQuietly(ofile);
    }
  } 
	 
  private double computeFMeasure(int [][] matrix){
    int [] classesDistribution = new int [matrix.length-1];  
	for(int i=0; i< matrix.length-1; i++){
      for(int j=0; j< matrix[i].length-1; j++){	          	          
	    classesDistribution[i]+=matrix[i][j];	
	  } 	        	   
	}    
	int posClassId = 0;
	int posNumInstances = classesDistribution[0]; 
	for (int k=1; k<matrix.length-1; k++) {
	  if (classesDistribution[k] < posNumInstances) {
	    posClassId = k;
	 	posNumInstances = classesDistribution[k];
	  }
	}
	double precision = 0.0;
	double recall = 0.0;	
	if(posClassId == 0){	  
	  precision = ((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));
	  recall = ((double)matrix[0][0]/(matrix[0][0]+matrix[1][0]));
	}
	else{	  
	  precision = ((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));
	  recall = ((double)matrix[1][1]/(matrix[1][1]+matrix[0][1]));
	}
	return ((2*precision*recall)/(recall+precision));
  }
  
  private double computeAuc(int [][] matrix){
    int [] classesDistribution = new int [matrix.length-1];  
	for(int i=0; i< matrix.length-1; i++){
      for(int j=0; j< matrix[i].length-1; j++){	          	          
	    classesDistribution[i]+=matrix[i][j];	
	  } 	        	   
	}    
	int posClassId = 0;
	int posNumInstances = classesDistribution[0]; 
	for (int k=1; k<matrix.length-1; k++) {
	  if (classesDistribution[k] < posNumInstances) {
	    posClassId = k;
	 	posNumInstances = classesDistribution[k];
	  }
	}
	double tp_rate = 0.0;
	double fp_rate = 0.0;
	if(posClassId == 0){
	  tp_rate=((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));	
	  fp_rate=((double)matrix[1][0]/(matrix[1][0]+matrix[1][1]));
	}
	else{
	  fp_rate=((double)matrix[0][1]/(matrix[0][1]+matrix[0][0]));	
	  tp_rate=((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));	
	}
	return ((1+tp_rate-fp_rate)/2);
  }
	  
  private double computeGM(int [][] matrix){
    int [] classesDistribution = new int [matrix.length-1];  
	for(int i=0; i< matrix.length-1; i++){
	  for(int j=0; j< matrix[i].length-1; j++){	          	          
	    classesDistribution[i]+=matrix[i][j];	
	  } 	        	   
	}    
	int posClassId = 0;
	int posNumInstances = classesDistribution[0]; 
	for (int k=1; k<matrix.length-1; k++) {
	  if (classesDistribution[k] < posNumInstances) {
	    posClassId = k;
		posNumInstances = classesDistribution[k];
	  }
	}
	double sensisivity = 0.0;
	double specificity = 0.0;
	if(posClassId == 0){
      sensisivity=((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));	
	  specificity=((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));
	}
	else{
	  specificity=((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));	
	  sensisivity=((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));	
	}
	return (Math.sqrt(sensisivity*specificity));  
  }
  
  private double[] computeSensisivityAndSpecificity(int [][] matrix){
    int [] classesDistribution = new int [matrix.length-1];  
	for(int i=0; i< matrix.length-1; i++){
	  for(int j=0; j< matrix[i].length-1; j++){	          	          
	    classesDistribution[i]+=matrix[i][j];	
	  } 	        	   
	}    
	int posClassId = 0;
	int posNumInstances = classesDistribution[0]; 
	for (int k=1; k<matrix.length-1; k++) {
	  if (classesDistribution[k] < posNumInstances) {
	    posClassId = k;
		posNumInstances = classesDistribution[k];
	  }
	}
	double sensisivity = 0.0;
	double specificity = 0.0;
	if(posClassId == 0){
      sensisivity=((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));	
	  specificity=((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));
	}
	else{
	  specificity=((double)matrix[0][0]/(matrix[0][0]+matrix[0][1]));	
	  sensisivity=((double)matrix[1][1]/(matrix[1][1]+matrix[1][0]));	
	}
	double []results = new double[2];
	results[0] = sensisivity;
	results[1] = specificity;
	return results;  
  }
	  
  private void sequential() throws IOException {

    log.info("Loading the forest...");
    DecisionForest forest = DecisionForest.load(getConf(), modelPath);

    if (forest == null) {
      log.error("No Decision Forest found!");
      return;
    }

    // load the dataset
    Dataset dataset = Dataset.load(getConf(), datasetPath);
    DataConverter converter = new DataConverter(dataset);

    log.info("Sequential classification...");
    long time = System.currentTimeMillis();

    Random rng = RandomUtils.getRandom();

    List<double[]> resList = new ArrayList<double[]>();
    if (dataFS.getFileStatus(dataPath).isDir()) {
      //the input is a directory of files
      testDirectory(outputPath, converter, forest, dataset, resList, rng);
    }  else {
      // the input is one single file
      testFile(dataPath, outputPath, converter, forest, dataset, resList, rng);
    }

    time = System.currentTimeMillis() - time;
    log.info("Classification Time: {}", DFUtils.elapsedTime(time));

    if (analyze) {
      if (dataset.isNumerical(dataset.getLabelId())) {
        RegressionResultAnalyzer regressionAnalyzer = new RegressionResultAnalyzer();
        double[][] results = new double[resList.size()][2];
        regressionAnalyzer.setInstances(resList.toArray(results));
        log.info("{}", regressionAnalyzer);
      } else {
        ResultAnalyzer analyzer = new ResultAnalyzer(Arrays.asList(dataset.labels()), "unknown");
        for (double[] r : resList) {
          analyzer.addInstance(dataset.getLabelString(r[0]),
            new ClassifierResult(dataset.getLabelString(r[1]), 1.0));
        }
        parseOutput(analyzer);
      }
    }
  }

  private void testDirectory(Path outPath,
                             DataConverter converter,
                             DecisionForest forest,
                             Dataset dataset,
                             Collection<double[]> results,
                             Random rng) throws IOException {
    Path[] infiles = DFUtils.listOutputFiles(dataFS, dataPath);

    for (Path path : infiles) {
      log.info("Classifying : {}", path);
      Path outfile = outPath != null ? new Path(outPath, path.getName()).suffix(".out") : null;
      testFile(path, outfile, converter, forest, dataset, results, rng);
    }
  }

  private void testFile(Path inPath,
                        Path outPath,
                        DataConverter converter,
                        DecisionForest forest,
                        Dataset dataset,
                        Collection<double[]> results,
                        Random rng) throws IOException {
    // create the predictions file
    FSDataOutputStream ofile = null;

    if (outPath != null) {
      ofile = outFS.create(outPath);
    }

    FSDataInputStream input = dataFS.open(inPath);
    try {
      Scanner scanner = new Scanner(input, "UTF-8");

      while (scanner.hasNextLine()) {
        String line = scanner.nextLine();
        if (line.isEmpty()) {
          continue; // skip empty lines
        }

        Instance instance = converter.convert(line);
        double prediction = forest.classify(dataset, rng, instance);

        if (ofile != null) {
          ofile.writeChars(Double.toString(prediction)); // write the prediction
          ofile.writeChar('\n');
        }
        
        results.add(new double[] {dataset.getLabel(instance), prediction});
      }

      scanner.close();
    } finally {
      Closeables.closeQuietly(input);
    }
  }

  private void writeToFileClassifyTime(String time) throws IOException{	
    FileSystem outFS = outputPath.getFileSystem(getConf());
	FSDataOutputStream ofile = null;		
	Path filenamePath = new Path(outputPath, dataName + "_classify_time").suffix(".txt");
	try    
	{	        	
	  if (ofile == null) {
	    // this is the first value, it contains the name of the input file
		ofile = outFS.create(filenamePath);
		// write the Classify Time	      	      	      	      
		StringBuilder returnString = new StringBuilder(200);	      
		returnString.append("=======================================================").append('\n');
		returnString.append("Classify Time\n");
		returnString.append("-------------------------------------------------------").append('\n');
		returnString.append(
			    		  StringUtils.rightPad(time,5)).append('\n');                  
		returnString.append("-------------------------------------------------------").append('\n');	      				
		String output = returnString.toString();
		ofile.writeUTF(output);
		ofile.close();		  
	  } 	    
	} 
	finally 
	{
      Closeables.closeQuietly(ofile);
	}
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new TestForest(), args);
  }

}