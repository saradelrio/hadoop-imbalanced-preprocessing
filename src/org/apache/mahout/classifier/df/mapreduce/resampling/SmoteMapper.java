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
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.mapreduce.SmoteBuilder;
import org.apache.mahout.classifier.df.resampling.tools.SMOTE;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

public class SmoteMapper extends Mapper<LongWritable, Text, LongWritable, Text>{

  private static final Logger log = LoggerFactory.getLogger(SmoteMapper.class);
  private Dataset dataset;
  boolean noOutput;
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  /** will contain all instances if this mapper's split */
  private final List<Instance> instances = Lists.newArrayList();

  protected void setup(Context context) throws IOException, InterruptedException {
	     
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    noOutput = !SmoteBuilder.isOutput(conf);
    
    dataset = SmoteBuilder.loadDataset(conf);
    
    converter = new DataConverter(dataset); 
  }
  
  public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    instances.add(converter.convert(value.toString()));	  
  }
  
  protected void cleanup (Context context) throws IOException, InterruptedException{
	   
    SMOTE smote;
    List<Instance> salida;
    
    smote = new SMOTE(dataset, instances, context);
    log.info("Executing SMOTE for each mapper's partition...");
    salida = smote.run();
    
    LongWritable key = new LongWritable(1);	
	   
    for(int i = 0 ; i < salida.size() ; i++){
      StringBuilder returnString = new StringBuilder();
      returnString.append(salida.get(i).toString(dataset)).append(dataset.getLabelString(dataset.getLabel(salida.get(i))));  
      String instance = returnString.toString();       
      Text text = new Text(instance);
      context.write(key, text);
    }
	
  }
  
}
