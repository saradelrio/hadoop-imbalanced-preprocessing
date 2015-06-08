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
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/*
 * An auxiliary class to initialize Instance Selection algorithms
 */
public class Metodo {
	
  protected static final Logger log = LoggerFactory.getLogger(Metodo.class);

  /*Data Structures*/
  protected Data training;  
  protected List<Instance> train_instances;
  protected Dataset dataset;
  
  /*Data Matrix*/
  protected double datosTrain[][];
  protected int clasesTrain[];
  
  /*Extra*/
  protected boolean nulosTrain[][];
  protected int nominalTrain[][];
  protected double realTrain[][];
  protected double minValues[];
  protected double maxValues[];
  
  protected boolean distanceEu;
  
  static protected double nominalDistance[][][];
  static protected double stdDev[];
  
  protected Context context;
  
  public Metodo(){}
  
  public Metodo(Dataset dataset, List<Instance> instances, Context context){
    
	this.train_instances = instances;
    this.dataset = dataset;
    this.training = new Data(dataset, instances);
    this.context = context;
    this.maxValues = training.getMaxValues();
    this.minValues = training.getMinValues();
    
    int nClases, i, j, l, m, n;
	double VDM;
	int Naxc, Nax, Nayc, Nay;
	double media, SD;		
	
	distanceEu = false;
	log.info("Normalizing instances of the mapper's partition...");
	try{
      normalizar(context);
    }catch (Exception e) {
      System.err.println(e);
      System.exit(1);
    }
	
    /*Previous computation for HVDM distance*/
	log.info("Previous computation for HVDM distance...");
    if (distanceEu == false) {    	
      stdDev = new double[dataset.nbAttributes()-1];
      nominalDistance = new double[dataset.nbAttributes()-1][][];
      nClases = dataset.nblabels();
                   
      for (i=0; i<nominalDistance.length; i++) {
        if (dataset.isCategorical(i)) {
          nominalDistance[i] = new double[dataset.nbValues(i)][dataset.nbValues(i)];
          for (j=0; j<dataset.nbValues(i); j++) { 
            nominalDistance[i][j][j] = 0.0;
          }
          for (j=0; j<dataset.nbValues(i); j++) {
            for (l=j+1; l<dataset.nbValues(i); l++) {
              VDM = 0.0;
              Nax = Nay = 0;
              for (m=0; m<training.size(); m++) {
                if (nominalTrain[m][i] == j) {
                  Nax++;
                }
                if (nominalTrain[m][i] == l) {
                  Nay++;
                }
              }
              for (m=0; m<nClases; m++) {
                Naxc = Nayc = 0;
                for (n=0; n<training.size(); n++) {
                  if (nominalTrain[n][i] == j && clasesTrain[n] == m) {
                    Naxc++;
                  }
                  if (nominalTrain[n][i] == l && clasesTrain[n] == m) {
                    Nayc++;
                  }
                }
                VDM += (((double)Naxc / (double)Nax) - ((double)Nayc / (double)Nay)) * (((double)Naxc / (double)Nax) - ((double)Nayc / (double)Nay));
              }
              nominalDistance[i][j][l] = Math.sqrt(VDM);
              nominalDistance[i][l][j] = Math.sqrt(VDM);
            }
          }
        } 
        else {
          media = 0;
          SD = 0;
          for (j=0; j<training.size(); j++) {
            media += realTrain[j][i];
            SD += realTrain[j][i]*realTrain[j][i];
          }
          media /= (double)realTrain.length;
          stdDev[i] = Math.sqrt((SD/((double)realTrain.length)) - (media*media));
        }	
      }
    }
  }
  
  /**
   * This function builds the data matrix for reference data and normalizes inputs values
   * @throws CheckException
   */
  protected void normalizar (Context context) throws CheckException {
	int i, k;
	
	datosTrain = new double[training.size()][dataset.nbAttributes()-1];
	clasesTrain = new int[training.size()];

	nulosTrain = new boolean[training.size()][dataset.nbAttributes()-1];
	nominalTrain = new int[training.size()][dataset.nbAttributes()-1];
	realTrain = new double[training.size()][dataset.nbAttributes()-1];

	for (i=0; i<training.size(); i++) {
	  log.info("Normalizing instance: "+i);
	  context.progress();
	  datosTrain[i] = training.get(i).get();		
	  clasesTrain[i] = (int) dataset.getLabel(training.get(i));	  
	  for (k = 0; k < datosTrain[i].length; k++) {	
		log.info("Normalizing instance: "+i);
		log.info("Normalizing att: "+k);
	    if (dataset.isCategorical(k)) {
		  nominalTrain[i][k] = (int)datosTrain[i][k]; 
		  datosTrain[i][k] /= dataset.nbValues(k) - 1;
		} else {
		  realTrain[i][k] = datosTrain[i][k];
		  //datosTrain[i][k] -= training.getMinAttribute(k);
		  datosTrain[i][k] -= minValues[k];
		  //datosTrain[i][k] /= training.getMaxAttribute(k) - training.getMinAttribute(k);
		  datosTrain[i][k] /= maxValues[k] - minValues[k];
		  if (Double.isNaN(datosTrain[i][k])){
		    datosTrain[i][k] = realTrain[i][k];
		  }
		}
	  }
	}            		
  } 
}
