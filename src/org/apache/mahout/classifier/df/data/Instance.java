/**
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

package org.apache.mahout.classifier.df.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.df.mapreduce.MapredOutput;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Represents one data instance.
 */
public class Instance implements Writable, Cloneable{
  
  /** attributes, except LABEL and IGNORED */
  private Vector attrs;
  
  public Instance() {
	
  }
  
  public Instance(Vector attrs) {
    this.attrs = attrs;
  }
  
  /**
   * Return the attribute at the specified position
   * 
   * @param index
   *          position of the attribute to retrieve
   * @return value of the attribute
   */
  public double get(int index) {
    return attrs.getQuick(index);
  }
  
  /**
   * Return all the input values.
   * @return a double[] with all input values
   */
  public double[] get() {
	int index;  
	double[] example = new double[attrs.size()-1];
	for(index = 0; index < attrs.size()-1 ; index++){
	  	example[index] = attrs.getQuick(index);
	}
	return example;
  }
  
  /**
   * Set the value at the given index
   * 
   * @param value
   *          a double value to set
   */
  public void set(int index, double value) {
    attrs.set(index, value);
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof Instance)) {
      return false;
    }
    
    Instance instance = (Instance) obj;
    
    return /*id == instance.id &&*/ attrs.equals(instance.attrs);
    
  }
  
  @Override
  public int hashCode() {
    return /*id +*/ attrs.hashCode();
  }
  
  public Instance clone() {
    return new Instance(attrs);
  }
  
  /*
  public String toString() {
	String string = new String();
	for(int i = 0 ; i < attrs.size()-1 ; i++){	
	  string += "" + attrs.getQuick(i) + ",";	  
	}
    return string;
  }*/
  
  public String toString(Dataset dataset) {
	String string = new String();
	for(int i = 0 ; i < attrs.size()-1 ; i++){	
	  if(dataset.isCategorical(i)){
		string += "" + dataset.getAttString(i, attrs.getQuick(i)) + ",";  
	  }else{
	    string += "" + attrs.getQuick(i) + ",";  
	  }	  	  
	}
    return string;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
	VectorWritable writable = new VectorWritable();
	writable.readFields(in);
	attrs = writable.get();	
  }

  @Override
  public void write(DataOutput out) throws IOException {
	VectorWritable vw = new VectorWritable(attrs);
	vw.setWritesLaxPrecision(true);
	vw.write(out);	
  }
}
