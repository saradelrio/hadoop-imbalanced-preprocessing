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
import org.apache.mahout.classifier.df.mapreduce.OversamplingBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class OversamplingMapper extends Mapper<LongWritable, Text, LongWritable, Text>{
  
  private static final Logger log = LoggerFactory.getLogger(OversamplingMapper.class);
  private Dataset dataset;
  boolean noOutput;
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  private String negativeClass;
  private int classes_distribution [];
  private int replication;
  
  protected void setup(Context context) throws IOException, InterruptedException {
     
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    noOutput = !OversamplingBuilder.isOutput(conf);
    
    dataset = OversamplingBuilder.loadDataset(conf);
       
    // The number of occurrences of each label value		  		     
    classes_distribution = new int [dataset.nblabels()];
    
    if(OversamplingBuilder.getNbNeg(conf) > OversamplingBuilder.getNbPos(conf)){    
      classes_distribution[0]= OversamplingBuilder.getNbNeg(conf);
      classes_distribution[1]= OversamplingBuilder.getNbPos(conf);
    }
    else{
	  classes_distribution[0]= OversamplingBuilder.getNbPos(conf);
      classes_distribution[1]= OversamplingBuilder.getNbNeg(conf);
    }
          
    negativeClass = OversamplingBuilder.getNegClass(conf);
    
    converter = new DataConverter(dataset); 
    
    double factor = (classes_distribution[0] / classes_distribution[1]); //ROS100
	
	double rand = Math.random();
	
	int integerPart = (int)Math.floor(factor);
	
	double decimalPart = factor - integerPart;
	
	if (rand < decimalPart)
		replication = integerPart + 1;
	else
		replication = integerPart;
  }      
  
  public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {	  
    Instance instance = converter.convert(value.toString());    
    int label_code = (int)dataset.getLabel(instance);    
    String label = dataset.getLabelString(label_code);
		
	LongWritable id;
	Random r = new Random();
	
	if (!noOutput) {  		
		if(label.equalsIgnoreCase(negativeClass)){
    		int random = r.nextInt(replication); 
    		id = new LongWritable(random);		
    		context.write(id, value);  
		}
		else{
			for(int i = 0 ; i < replication ; i++){   
				id = new LongWritable(i);
				context.write(id, value);  
			}
		}	
	}    	  
  }
}
