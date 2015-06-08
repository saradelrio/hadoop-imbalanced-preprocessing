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

public class MTwister {

  long[] state=new long[624];
  int left = 1;
  int initf = 0;
  int inext;

  public MTwister() {}

  public MTwister(long s) {
	init_genrand(s);
  }
 	
  public MTwister(long[] init_key) {
    init_by_array(init_key);
  }

  public void init_genrand(long s) {
	int j;
	state[0]= s & 0xffffffffL;
	for (j=1; j<624; j++) {
	  state[j] = (1812433253L * (state[j-1] ^ (state[j-1] >> 30)) + j);
	  state[j] &= 0xffffffffL;
    }
	left = 1;
	initf = 1;
  }

  void init_by_array(long[] init_key) {
	int i, j, k;
	int key_length;
	key_length = init_key.length;
	init_genrand(19650218L);
	i=1; j=0;
	k = (624>key_length ? 624 : key_length);
	for (; k>0; k--) {
	  state[i] = (state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1664525L))+ init_key[j] + j;
	  state[i] &= 0xffffffffL;
	  i++; j++;
	  if (i>=624) {
		state[0] = state[624 -1];
		i=1; 
	  }
	  if (j>=key_length) 
		j=0;
	}
    for (k=624 -1; k>0; k--) {
	  state[i] = (state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1566083941L)) - i;
	  state[i] &= 0xffffffffL;
	  i++;
	  if (i>=624) { 
	    state[0] = state[624 -1]; i=1; 
	  }
    }
    state[0] = 0x80000000L;
    left = 1;
    initf = 1;
  }

  void next_state() {
	int ip=0;
	int j;
	if (initf==0) init_genrand(5489L);
	  left = 624;
	  inext = 0;
	  for (j=624 -397 +1; (--j)>0; ip++)
		state[ip] = state[ip+397] ^ ((( ((state[ip+0]) & 0x80000000L) | ((state[ip+1]) & 0x7fffffffL) ) >> 1) ^ (((state[ip+1]) & 1L) != 0L ? 0x9908b0dfL : 0L));
	  for (j=397; (--j)>0; ip++)
		state[ip] = state[ip+397 -624] ^ ((( ((state[ip+0]) & 0x80000000L) | ((state[ip+1]) & 0x7fffffffL) ) >> 1) ^ (((state[ip+1]) & 1L) != 0L ? 0x9908b0dfL : 0L));
	  state[ip] = state[ip+397 -624] ^ ((( ((state[ip+0]) & 0x80000000L) | ((state[0]) & 0x7fffffffL) ) >> 1) ^ (((state[0]) & 1L) != 0L ? 0x9908b0dfL : 0L));
  }

  // generates a random number on [0,0xffffffff]-interval 
  long genrand_int32() {
	long y;
	if (--left == 0) next_state();
	y = state[inext++];
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680L;
	y ^= (y << 15) & 0xefc60000L;
	y ^= (y >> 18);
	return y;
  }

  // generates a random number on [0,0x7fffffff]-interval 
  long genrand_int31() {
	long y;
	if (--left == 0) next_state();
	y = state[inext++];
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680L;
	y ^= (y << 15) & 0xefc60000L;
	y ^= (y >> 18);
	return (long)(y>>1);
  }

  // generates a random number on [0,1]-real-interval 
  public double genrand_real1() {
	long y;
	if (--left == 0) next_state();
	y = state[inext++];
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680L;
	y ^= (y << 15) & 0xefc60000L;
	y ^= (y >> 18);
	return (double)y * (1.0/4294967295.0);
  }

  // generates a random number on [0,1)-real-interval 
  public double genrand_real2() {
	long y;
	if (--left == 0) next_state();
	y = state[inext++];
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680L;
	y ^= (y << 15) & 0xefc60000L;
	y ^= (y >> 18);
	return (double)y * (1.0/4294967296.0);
  }

  // generates a random number on (0,1)-real-interval 
  public double genrand_real3() {
	long y;
	if (--left == 0)
		next_state();
	y = state[inext++];
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680L;
	y ^= (y << 15) & 0xefc60000L;
	y ^= (y >> 18);
	return ((double)y + 0.5) * (1.0/4294967296.0);
  }

  // generates a random number on [0,1) with 53-bit resolution
  public double genrand_res53() {
	long a=genrand_int32()>>5, b=genrand_int32()>>6;
	return(a*67108864.0+b)*(1.0/9007199254740992.0);
  }

  // generates a standardized gaussian random number 
  public double genrand_gaussian() {
	int i;
	double a;
	a=0.0;
	for(i=0; i<6; i++) {
		a += genrand_real1();
		a -= genrand_real1();
	}
	return a;
  }

  // returns the state of the random number generator 
  public long[] getState() {
	int i;
	long[] savedState=new long[627];
	for(i=0; i<624; i++) savedState[i] = state[i];
	savedState[624] = (long) left;
	savedState[625] = (long) initf;
	savedState[626] = (long) inext;
	return savedState;
  }

  // restores the state of the random number generator 
  public void setState(long[] savedState) {
	int i;
	for(i=0; i<624; i++) state[i] = savedState[i];
	left = (int) savedState[624];
	initf = (int) savedState[625];
	inext = (int) savedState[626];
  }
}
