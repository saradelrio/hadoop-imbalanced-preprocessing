/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.classifier.df.resampling.tools;

import java.util.List;

import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;

import com.google.common.collect.Lists;

public class SMOTE extends Metodo{

  private long semilla;
  private int kSMOTE = 5;
  
  public SMOTE(Dataset dataset, List<Instance> instances, Context context){
    super(dataset, instances, context);
  }
  
  public List<Instance> run(){
	  
    int nPos = 0;
    int nNeg = 0;
    int i, j, l, m, k;
    int tmp;
    int posID, negID;
    int positives[];
    int negatives[];
    double conjS[][];
    double conjR[][];
    int conjN[][];
    boolean conjM[][];
    int clasesS[];
    double genS[][];
	double genR[][];
	int genN[][];
	boolean genM[][];
    int clasesGen[];
    int tamS;
    int pos;
    int neighbors[][];
    int nn;
    
    /*Count of number of positive and negative examples*/ 
    log.info("Count of number of positive and negative examples...");
    int classes_distribution [] = training.computeClassDistribution();
    int negative_class = training.computeNegativeClass(classes_distribution);
	  
    nPos = classes_distribution [0];
    nNeg = classes_distribution [1];
    
    if (nPos > nNeg) {
      tmp = nPos;
      nPos = nNeg;
      nNeg = tmp;
      posID = 1;
      negID = 0;
    } else {
      posID = 0;
      negID = 1;
    }
    
    log.info("Number of positive examples: ", nPos);
    log.info("Number of negative examples: ", nNeg);
    
    /*Localize the positive and the negative instances*/
    log.info("Localize the positive and the negative instances...");
    List<Instance> instances = training.getInstances();
    positives = new int[nPos];
    negatives = new int[nNeg];
    for (i=0, j=0, k =0; i < training.size() ; i++) {
      context.progress();
      if ((int)dataset.getLabel(instances.get(i)) == posID) {
        positives[j] = i;
        j++;
      }
      else{
    	negatives[k] = i;  
    	k++;
      }
    }
    
    if(nPos > 0){    
      /*Randomize the instance presentation*/
      log.info("Randomize the instance presentation...");
      Randomize.setSeed (semilla);
      for (i=0; i<positives.length; i++) {
        context.progress();
        tmp = positives[i];
        pos = Randomize.Randint(0,positives.length-1);
        positives[i] = positives[pos];
        positives[pos] = tmp;
      }
    
      /*Obtain k-nearest neighbors of each positive instance*/
      log.info("Obtain k-nearest neighbors of each positive instance...");
      neighbors = new int[positives.length][kSMOTE];
      for (i=0; i<positives.length; i++) {
        log.info("Obtain k-nearest neighbors of each positive instance: "+i);
        context.progress();
        KNN.evaluacionKNN2 (kSMOTE, datosTrain, realTrain, nominalTrain, nulosTrain, clasesTrain, datosTrain[positives[i]], 
    		  realTrain[positives[i]], nominalTrain[positives[i]], nulosTrain[positives[i]], 2, distanceEu, neighbors[i], dataset, context);
      }
    
      /*Interpolation of the minority instances*/
      // balance = true
      log.info("Interpolation of the minority instances...");
      genS = new double[nNeg-nPos][datosTrain[0].length];
	  genR = new double[nNeg-nPos][datosTrain[0].length];
	  genN = new int[nNeg-nPos][datosTrain[0].length];
	  genM = new boolean[nNeg-nPos][datosTrain[0].length];
	  clasesGen = new int[nNeg-nPos];
	  
	  for (i=0; i<genS.length; i++) {
	    context.progress();
	    clasesGen[i] = posID;
	    nn = Randomize.Randint(0,kSMOTE-1);
	    double ra[] = realTrain[positives[i%positives.length]];
	    double rb[] = realTrain[neighbors[i%positives.length][nn]];
	    int na[] = nominalTrain[positives[i%positives.length]];
	    int nb[] = nominalTrain[neighbors[i%positives.length][nn]];
	    boolean ma[] = nulosTrain[positives[i%positives.length]];
	    boolean mb[] = nulosTrain[neighbors[i%positives.length][nn]];
	    interpolate (ra,rb,na,nb,ma,mb,genS[i],genR[i],genN[i],genM[i]);
      }
	
	  // balance = true
      tamS = 2*nNeg;
    
      /*Construction of the S set from the previous vector S*/
      log.info("Construction of the S set from the previous vector S...");
      conjS = new double[tamS][datosTrain[0].length];
      conjR = new double[tamS][datosTrain[0].length];
      conjN = new int[tamS][datosTrain[0].length];
      conjM = new boolean[tamS][datosTrain[0].length];
      clasesS = new int[tamS];
      for (j=0; j<datosTrain.length; j++) {
        context.progress();
        for (l=0; l<datosTrain[0].length; l++) {
    	  context.progress();
          conjS[j][l] = datosTrain[j][l];
          conjR[j][l] = realTrain[j][l];
          conjN[j][l] = nominalTrain[j][l];
          conjM[j][l] = nulosTrain[j][l];
        }
        clasesS[j] = clasesTrain[j];
      }
      for (m=0;j<tamS; j++, m++) {
        context.progress();
        for (l=0; l<datosTrain[0].length; l++) {
    	  context.progress();
          conjS[j][l] = genS[m][l];
          conjR[j][l] = genR[m][l];
          conjN[j][l] = genN[m][l];
          conjM[j][l] = genM[m][l];
        }
        clasesS[j] = clasesGen[m];
      }
      log.info("Writing output...");
      return escribeSalida(conjR, conjN, conjM, clasesS, dataset, negative_class, negatives, context);        
    }
    else{
      return escribeSalida(negatives, context);
    }
  }
  
  /**
   * <p>
   * Generates a synthetic example for the minority class from two existing
   * examples in the current population
   * </p>
   *
   * @param ra    Array with the real values of the first example in the current population
   * @param rb    Array with the real values of the second example in the current population
   * @param na    Array with the nominal values of the first example in the current population
   * @param nb    Array with the nominal values of the second example in the current population
   * @param ma    Array with the missing values of the first example in the current population
   * @param mb    Array with the missing values of the second example in the current population
   * @param resS  Array with the general data about the generated example
   * @param resR  Array with the real values of the generated example
   * @param resN  Array with the nominal values of the generated example
   * @param resM  Array with the missing values of the generated example
   * @param interpolation Kind of interpolation used to generate the synthetic example
   * @param alpha Parameter used in the BLX-alpha interpolation
   * @param mu    Parameter used in the SBX interpolation and the normal interpolation
   */
  void interpolate (double ra[], double rb[], int na[], int nb[], boolean ma[], boolean mb[], double resS[], double resR[], int resN[], boolean resM[]) {
    int i;
    double diff;
    double gap;
    int suerte;

    gap = Randomize.Rand();
    for (i=0; i<ra.length; i++) {
      context.progress();
      resM[i] = false;
      if (!dataset.isCategorical(i)) {
        //interpolation.equalsIgnoreCase("standard")       
        diff = rb[i] - ra[i];
        resR[i] = ra[i] + gap*diff;
        //resS[i] = (resR[i] + training.getMinAttribute(i)) / (training.getMaxAttribute(i) - training.getMinAttribute(i));          
        resS[i] = (resR[i] + minValues[i]) / (maxValues[i] - minValues[i]);
      }
      else{
        suerte = Randomize.Randint(0, 2);
        if (suerte == 0) {
          resN[i] = na[i];
        } 
        else {
          resN[i] = nb[i];
        }
        resS[i] = (double)resN[i] / (double)(dataset.nbValues(i) - 1);
      }
    }
  }
  
  /**
   * <p>
   * Computes the triangular distance between three values
   * </p>
   *
   * @param A First value used to compute the triangular distance
   * @param B Second value used to compute the triangular distance
   * @param C Third value used to compute the triangular distance
   * @return the triangular distance between the three values
   */
  double DistTriangular(double A, double B, double C) {
    double S, T, Z, u_1, u_2, x;

    if (A==C) return(B);
    S=B-A;
    T=C-A;
    u_1 = Randomize.Rand();
    u_2 = Randomize.Rand();
    Z = S/T;
    if (u_1<= Z) {
      x=S*Math.sqrt(u_2)+A;
      return(x);
    } else {
      x=T-(T-S)*Math.sqrt(u_2)+A;
      return(x);
    }
  }

  /**
   * <p>
   * Obtains a normal value transformation from a given data
   * </p>
   *
   * @param desv  Value that is going to be transformed
   * @return  the normal value transformation applied
   */
  double NormalValue (double desv) {
    double u1, u2;

    u1=Randomize.Rand();
    u2=Randomize.Rand();

    return desv * Math.sqrt (-2*Math.log(u1)) * Math.sin (2*Math.PI*u2);
  }
  

	/**
	 * Writes results (Required for HVDM distance)
	 *
	 * @param nombreFichero Name of the output file
	 * @param realIN Real values of instances to write
	 * @param nominalIN Nominal values of instances to write
	 * @param nulosIN NUll values of instances to write
	 * @param instanciasOUT Instances to mantain
	 * @param dataset dataset description
	 *//*
	public List<Instance> escribeSalida (double realIN[][], int nominalIN[][], boolean nulosIN[][], int instanciasOUT[], Dataset dataset) {
	  List<Instance> instances = Lists.newArrayList();
	  DataConverter converter = new DataConverter(dataset);
	  String cadena = "";
	  int i, j;

	  for (i=0; i<realIN.length; i++){
	    cadena = "";
	    for (j=0; j<realIN[i].length; j++) {
	      if (nulosIN[i][j] == false) {					  
	        if (dataset.isNumerical(j)) {
		      cadena += String.valueOf(realIN[i][j]) + ",";					
		    } 
	        else {
	          cadena += dataset.getAttString(j,nominalIN[i][j]) + ",";	
		    }
		  } 
	      else {				  
	        cadena += "?,";
		  }
	    }
	    if (dataset.isNumerical(dataset.getLabelId())) {
	      cadena += String.valueOf(instanciasOUT[i]);
	    } 
	    else {
	      cadena += dataset.getLabelString(instanciasOUT[i]);
	    }
	    instances.add(converter.convert(cadena));	  
	  }		
	  return instances;
	}*/
  
  public List<Instance> escribeSalida (double realIN[][], int nominalIN[][], boolean nulosIN[][], int instanciasOUT[], Dataset dataset, int negative_class, int negatives[], Context context) {
    List<Instance> instances = Lists.newArrayList();
    DataConverter converter = new DataConverter(dataset);
    String cadena = "";
    int i, j;
    //positive instances
    for (i=0; i<realIN.length; i++){
      context.progress();
      if(instanciasOUT[i] != negative_class){
        cadena = "";
        for (j=0; j<realIN[i].length; j++) {
          if (nulosIN[i][j] == false) {					  
            if (dataset.isNumerical(j)) {
	          cadena += String.valueOf(realIN[i][j]) + ",";					
	        } 
            else {
              cadena += dataset.getAttString(j,nominalIN[i][j]) + ",";	
	        }
	      } 
          else {				  
            cadena += "?,";
	      }
        }
        if (dataset.isNumerical(dataset.getLabelId())) {
          cadena += String.valueOf(instanciasOUT[i]);
        } 
        else {
          cadena += dataset.getLabelString(instanciasOUT[i]);
        }
        instances.add(converter.convert(cadena));	  
      }
    }
    //negative instances
    for(i=0; i <negatives.length; i++){
      context.progress();
      instances.add(train_instances.get(negatives[i]));	
    }
    return instances;
  }
  
  public List<Instance> escribeSalida (int negatives[], Context context) {
    List<Instance> instances = Lists.newArrayList();
    //negative instances
    for(int i=0; i <negatives.length; i++){
      context.progress();
      instances.add(train_instances.get(negatives[i]));	
    }
    return instances;
  }
 
}
