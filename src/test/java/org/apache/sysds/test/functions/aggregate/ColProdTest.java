/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.aggregate;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

public class ColProdTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ColProdTest.class.getSimpleName() + "/";
	private final static String TEST_GENERAL = "General";
	private final static String TEST_SCALAR = "Scalar";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_GENERAL,
			new TestConfiguration(TEST_CLASS_DIR, "ColProdTest", new String[] {"vector_colprod", "matrix_colprod"}));
	}

	@Test
	public void testGeneral() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = getTestConfiguration(TEST_GENERAL);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		loadTestConfiguration(config);

		double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		double[][] vectorColProd = new double[1][1];
		for(int i = 0; i < rows; i++) {
			vectorColProd[0][0] *= vector[i][0];
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_colprod", vectorColProd);

		double[][] matrix = getRandomMatrix(rows, cols, 0, 1, 1, -1);
		double[][] matrixColProd = new double[1][1];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				matrixColProd[0][j] *= matrix[i][j];
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_colprod", matrixColProd);

		runTest();
		compareResults(1e-14);
	}
}
