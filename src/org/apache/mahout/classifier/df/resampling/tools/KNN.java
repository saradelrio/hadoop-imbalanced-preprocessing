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

import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.classifier.df.data.Dataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KNN {
  private static final Logger log = LoggerFactory.getLogger(KNN.class);
 
  /** 
   * Calculates the Euclidean distance between two instances
   * 
   * @param instance1 First instance 
   * @param instance2 Second instance
   * @return The Euclidean distance
   * 
   */
  protected static double distance(double instance1[],double instance2[]){
	
    double length=0.0;

	for (int i=0; i<instance1.length; i++) {
	  length += (instance1[i]-instance2[i])*(instance1[i]-instance2[i]);
	}
		
	length = Math.sqrt(length); 
			
	return length;
	
  } 
    
         
  /** 
   * Evaluates a instance to predict its class.
   * 
   * @param example Instance evaluated 
   * @return Class predicted
   * 
   */
  public static int evaluate (double example[], double trainData[][],int nClasses,int trainOutput[],int k) {

	double minDist[];
	int nearestN[];
	int selectedClasses[];
	double dist;
	int prediction;
	int predictionValue;
	boolean stop;

	nearestN = new int[k];
	minDist = new double[k];

    for (int i=0; i<k; i++) {
      nearestN[i] = 0;
	  minDist[i] = Double.MAX_VALUE;
	}
	
    //KNN Method starts here
    
	for (int i=0; i<trainData.length; i++) {
	
      dist = distance(trainData[i],example);

	  if (dist > 0.0){ //leave-one-out
	
		//see if it's nearer than our previous selected neighbors
		stop=false;
		
		for(int j=0;j<k && !stop;j++){
		
		  if (dist < minDist[j]) {
		    
			for (int l = k - 1; l >= j+1; l--) {
			  minDist[l] = minDist[l - 1];
			  nearestN[l] = nearestN[l - 1];
			}	
			
			minDist[j] = dist;
			nearestN[j] = i;
			stop=true;
		  }
		}
	  }
	}
	
	//we have check all the instances... see what is the most present class
	selectedClasses= new int[nClasses];

	for (int i=0; i<nClasses; i++) {
	  selectedClasses[i] = 0;
	}	
	
	for (int i=0; i<k; i++) {
      selectedClasses[trainOutput[nearestN[i]]]+=1;
	}
	
	prediction=0;
	predictionValue=selectedClasses[0];
	
	for (int i=1; i<nClasses; i++) {
      if (predictionValue < selectedClasses[i]) {
        predictionValue = selectedClasses[i];
        prediction = i;
      }
	}
	
	return prediction;

  } 	

  /** 
   * Calculates the Euclidean distance between two instances
   * 
   * @param ej1 First instance 
   * @param ej2 Second instance
   * @return The Euclidean distance
   * 
   */
  public static double distancia (double ej1[], double ej2[]) {

	int i;
	double suma = 0;

	for (i=0; i<ej1.length; i++) {
	  suma += (ej1[i]-ej2[i])*(ej1[i]-ej2[i]);
	}
	suma = Math.sqrt(suma);

	return suma;
	  
  } 

  /** 
   * Calculates the unsquared Euclidean distance between two instances
   * 
   * @param ej1 First instance 
   * @param ej2 Second instance
   * @return The unsquared Euclidean distance
   * 
   */
  public static double distancia2 (double ej1[], double ej2[]) {

	int i;
	double suma = 0;

	for (i=0; i<ej1.length; i++) {
	  suma += (ej1[i]-ej2[i])*(ej1[i]-ej2[i]);
	}
	return suma;  
	  
  } 

  /**
   * Executes KNN
   *
   * @param nvec Number of neighbors
   * @param conj Reference to the training set
   * @param real  Reference to the training set (real valued)
   * @param nominal  Reference to the training set (nominal valued)	 
   * @param nulos  Reference to the training set (null values)	
   * @param clases Output attribute of each instance
   * @param ejemplo New instance to classifiy
   * @param ejReal New instance to classifiy	 (real valued)
   * @param ejNominal New instance to classifiy	 (nominal valued)	
   * @param ejNulos New instance to classifiy	 (null values)	
   * @param nClases Number of classes of the problem
   * @param distance True= Euclidean distance; False= HVDM
   * @param vecinos Neighbors of the new instance
   *
   * @return Class of the new instance
   */	   
  public static int evaluacionKNN2 (int nvec, double conj[][], double real[][], int nominal[][], boolean nulos[][], int clases[], double ejemplo[], double ejReal[], int ejNominal[], boolean ejNulos[], int nClases, boolean distance, int vecinos[], Dataset dataset, Context context) {

    int i, j, l;
    boolean parar = false;
    int vecinosCercanos[];
    double minDistancias[];
    int votos[];
    double dist;
    int votada, votaciones;

    if (nvec > conj.length)
	  nvec = conj.length;
    votos = new int[nClases];
    vecinosCercanos = new int[nvec];
    minDistancias = new double[nvec];
    for (i=0; i<nvec; i++) {
	  vecinosCercanos[i] = -1;
	  minDistancias[i] = Double.POSITIVE_INFINITY;
    }

    for (i=0; i<conj.length; i++) {
      context.progress();	
	  dist = distancia(conj[i], real[i], nominal[i], nulos[i], ejemplo, ejReal, ejNominal, ejNulos, distance, dataset);
	  if (dist > 0) {
	    parar = false;
	    for (j = 0; j < nvec && !parar; j++) {
	      context.progress();	
		  if (dist < minDistancias[j]) {
		    parar = true;
		    for (l = nvec - 1; l >= j+1; l--) {
		      context.progress();
			  minDistancias[l] = minDistancias[l - 1];
			  vecinosCercanos[l] = vecinosCercanos[l - 1];
		    }
		    minDistancias[j] = dist;
		    vecinosCercanos[j] = i;
		  }
	    }
	  }
    }

    for (j=0; j<nClases; j++) {
	  votos[j] = 0;
    }
    for (j=0; j<nvec; j++) {
	  if (vecinosCercanos[j] >= 0)
		votos[clases[vecinosCercanos[j]]] ++;
    }
    votada = 0;
    votaciones = votos[0];
    for (j=1; j<nClases; j++) {
	  if (votaciones < votos[j]) {
	    votaciones = votos[j];
	    votada = j;
	  }
    }

    for (i=0; i<vecinosCercanos.length; i++)
	  vecinos[i] = vecinosCercanos[i];
  
    return votada;
	  
  } 

  /** 
   * Calculates the HVDM distance between two instances
   * 
   * @param ej1 First instance 
   * @param ej1Real First instance (Real valued)	 
   * @param ej1Nom First instance (Nominal valued)	
   * @param ej1Nul First instance (Null values)		 
   * @param ej2 Second instance
   * @param ej2Real First instance (Real valued)	 
   * @param ej2Nom First instance (Nominal valued)	
   * @param ej2Nul First instance (Null values)	
   * @param Euc Use euclidean distance instead of HVDM
   *
   * @return The HVDM distance 
   */
  public static double distancia (double ej1[], double ej1Real[], int ej1Nom[], boolean ej1Nul[], double ej2[], double ej2Real[], int ej2Nom[], boolean ej2Nul[], boolean Euc, Dataset dataset) {

    int i;
    double suma = 0;
    if (Euc == true) {
	  for (i=0; i<ej1.length; i++) {
		suma += (ej1[i]-ej2[i])*(ej1[i]-ej2[i]);
	  }
	  suma = Math.sqrt(suma);    
    } else {
	  for (i=0; i<ej1.length; i++) {
	    if (ej1Nul[i] == true || ej2Nul[i] == true) {
		  suma += 1;
	    } else if (dataset.isCategorical(i)) {
		  suma += Metodo.nominalDistance[i][ej1Nom[i]][ej2Nom[i]];
	    } else {
		  suma += Math.abs(ej1Real[i]-ej2Real[i]) / 4*Metodo.stdDev[i];
	    }
	  }
	  suma = Math.sqrt(suma);       	
    }

    return suma;
	  
  } 
  
}
