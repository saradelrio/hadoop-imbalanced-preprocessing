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
package org.apache.mahout.classifier.df.mapreduce.resampling;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.mapreduce.UndersamplingBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UndersamplingMapper extends Mapper<LongWritable, Text, LongWritable, Text>{

  private static final Logger log = LoggerFactory.getLogger(UndersamplingMapper.class);
  private Dataset dataset;
  boolean noOutput;
  private String positiveClass;
  private int classes_distribution [];
  private double elimination_factor;
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  protected void setup(Context context) throws IOException, InterruptedException {
	     
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    noOutput = !UndersamplingBuilder.isOutput(conf);
    
    dataset = UndersamplingBuilder.loadDataset(conf);
    
    converter = new DataConverter(dataset); 
    
    // The number of occurrences of each label value		  		     
    classes_distribution = new int [dataset.nblabels()];
	
	if(UndersamplingBuilder.getNbNeg(conf) > UndersamplingBuilder.getNbPos(conf)){    
	  classes_distribution[0]= UndersamplingBuilder.getNbNeg(conf);
	  classes_distribution[1]= UndersamplingBuilder.getNbPos(conf);
	}
	else{
	  classes_distribution[0]= UndersamplingBuilder.getNbPos(conf);
	  classes_distribution[1]= UndersamplingBuilder.getNbNeg(conf);
	}
            
    positiveClass = UndersamplingBuilder.getPosClass(conf);
    
    elimination_factor = (double)classes_distribution[1] / (double)classes_distribution[0]; //50-50
  }      
  
  public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {    
    Instance instance = converter.convert(value.toString());	   
    int label_code = (int)dataset.getLabel(instance);  
    String label = dataset.getLabelString(label_code);
    
    Random r = new Random();
    double random;
        
    if (!noOutput) {  		
		if(label.equalsIgnoreCase(positiveClass)){		
    		context.write(key, value);  
		}
		else{//negative class
			random = r.nextDouble(); 
			if(random  < elimination_factor){   
				context.write(key, value);  
			}
		}	
	}    	  
	
  }
  
  
	  
}

