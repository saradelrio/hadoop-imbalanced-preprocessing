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

public class Referencia implements Comparable {

  //values of the reference
  public int entero;
  public double real;

  /**
   * Default builder
   */
  public Referencia () {} 

  /**
   * Builder
   *
   * @param a Integer value
   * @param b Double value
   */
  public Referencia (int a, double b) {
    entero = a;
	real = b;
  }

  /**
   * Compare to Method
   *
   * @param o1 Reference to compare
   *
   * @return Relative order between the references
   */
  public int compareTo (Object o1) {
	if (this.real > ((Referencia)o1).real)
	  return -1;
	else if (this.real < ((Referencia)o1).real)
	  return 1;
	else return 0;
  }

  /**
   * To String Method
   *
   * @return String representation of the chromosome
   */
  public String toString () {
	return new String ("{"+entero+", "+real+"}");
  }
}
