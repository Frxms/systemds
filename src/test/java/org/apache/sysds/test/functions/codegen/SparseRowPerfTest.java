package org.apache.sysds.test.functions.codegen;

import org.apache.hadoop.util.StringUtils;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.InputType;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;

import static org.apache.sysds.runtime.codegen.LibSpoofPrimitives.vectDivWrite;
import static org.apache.sysds.runtime.codegen.LibSpoofPrimitives.vectMinWrite;
import static org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.getOutputType;
import static org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.getTestType;

public class SparseRowPerfTest {

	private final int m;
	private final int n;
	private final double sparsity2;
	private final int warmupRuns;
	private final int repetitions;

	private static final double sparsity1 = 0.9;

	public SparseRowPerfTest() {
		this(20, 10000, 50, 5000, 0.08);
	}

	public SparseRowPerfTest(int rl, int cl, int warmupRuns, int repetitions, double sparsity) {
		m = rl;
		n = cl;
		this.warmupRuns = warmupRuns;
		this.repetitions = repetitions;
		this.sparsity2 = sparsity;
	}

	private void testSparseBinaryPrimitivesPerf() {
		long sparseResults[] = new long[repetitions];
		long denseResults[] = new long[repetitions];
		for(int i = 0; i < warmupRuns; i++) {
			runSparseBinaryPrimitvesTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
			runDenseBinaryPrimitvesTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
		}
		for(int i = 0; i < repetitions; i++) {
			sparseResults[i] = runSparseBinaryPrimitvesTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
			denseResults[i] = runDenseBinaryPrimitvesTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
		}
		System.out.println("Sparsity: " + sparsity2 + "; rl: " + m + "; cl: " + n);
		System.out.println("Via reflection:");
		System.out.println("Sparse calculation: " + Arrays.stream(sparseResults).average().getAsDouble() + " nanos");
		System.out.println("Dense calculation: " + Arrays.stream(denseResults).average().getAsDouble() + " nanos");
		for(int i = 0; i < warmupRuns; i++) {
			runSparseDivTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
			runDenseDivTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
		}
		for(int i = 0; i < repetitions; i++) {
			sparseResults[i] = runSparseDivTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
			denseResults[i] = runDenseDivTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
		}
		System.out.println("Via direct method call:");
		System.out.println("Sparse calculation: " + Arrays.stream(sparseResults).average().getAsDouble() + " nanos");
		System.out.println("Dense calculation: " + Arrays.stream(denseResults).average().getAsDouble() + " nanos");
		timingUtilSparsePrimitivesTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
		timingUtilDensePrimitivesTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	private long runSparseDivTest(BinType binType, InputType inputType1, InputType inputType2) {
		double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 1264);
		double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 1265);
		long nanos = System.nanoTime();
		for(int i = 0; i < m; i++) {
			vectDivWrite(n, inA.getSparseBlock().values(i), inB.max(),
				inA.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i));
		}
		return System.nanoTime() - nanos;
	}

	private long runDenseDivTest(BinType binType, InputType inputType1, InputType inputType2) {
		double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 1264);
		double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 1265);
		long nanos = System.nanoTime();
		for(int i = 0; i < m; i++) {
			vectDivWrite(inA.getSparseBlock().values(i), inB.max(),
				inA.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i), n);
		}
		return System.nanoTime() - nanos;
	}

	private long runSparseBinaryPrimitvesTest(BinType binType, InputType inputType1, InputType inputType2) {
		try {
			double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 1264);
			double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 1265);

			String meName = "vect"+ StringUtils.camelize(binType.name().split("_")[1])+"Write";
			final Method me;
			if( inputType1==InputType.VECTOR_SPARSE && inputType2==InputType.SCALAR )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{int.class, double[].class, double.class, int[].class, int.class, int.class});
			else if( inputType1==InputType.SCALAR && inputType2==InputType.VECTOR_SPARSE )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{int.class, double.class, double[].class, int[].class, int.class, int.class});
			else //if( type1==InputType.VECTOR_SPARSE && type2==InputType.VECTOR_SPARSE )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{int.class, double[].class, double[].class, int[].class, int[].class, int.class, int.class, int.class, int.class});

			long nanos;
			if(inputType1 == InputType.VECTOR_SPARSE && inputType2 == InputType.SCALAR) {
				 nanos = System.nanoTime();
				for(int i = 0; i < m; i++) {
					me.invoke(null, n, inA.getSparseBlock().values(i), inB.max(),
						inA.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i));
				}
				return System.nanoTime() - nanos;
			}else if(inputType1 == InputType.SCALAR && inputType2 == InputType.VECTOR_SPARSE) {
				nanos = System.nanoTime();
				for(int i = 0; i < m; i++) {
					me.invoke(null, n, inA.max(), inB.getSparseBlock().values(i),
						inB.getSparseBlock().indexes(i), inB.getSparseBlock().pos(i), inB.getSparseBlock().size(i));
				}
				return System.nanoTime() - nanos;
			}else if(inputType1 == InputType.VECTOR_SPARSE && inputType2 == InputType.VECTOR_SPARSE) {
				nanos = System.nanoTime();
				for(int i = 0; i < m; i++) {
					me.invoke(null, n, inA.getSparseBlock().values(i),
						inB.getSparseBlock().values(i), inA.getSparseBlock().indexes(i),
						inB.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inB.getSparseBlock().pos(i),
						inA.getSparseBlock().size(i), inB.getSparseBlock().size(i));
				}
				return System.nanoTime() - nanos;
			}
				return Long.MAX_VALUE;
		}catch(Exception ex) {
				throw new RuntimeException(ex);
			}
		}

	private long runDenseBinaryPrimitvesTest(BinType binType, InputType inputType1, InputType inputType2) {
		try {
			//generate input data (scalar later derived if needed)
			double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 3);
			double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 7);

			//get vector primitive via reflection
			String meName = "vect" + StringUtils.camelize(binType.name().split("_")[1]) + "Write";
			final Method me;
			if(inputType1 == InputType.SCALAR && inputType2 == InputType.VECTOR_DENSE)
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double.class, double[].class, int.class, int.class});
			else if(inputType1 == InputType.VECTOR_DENSE && inputType2 == InputType.SCALAR)
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double[].class, double.class, int.class, int.class});
			else if(inputType1 == InputType.VECTOR_DENSE && inputType2 == InputType.VECTOR_DENSE)
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double[].class, double[].class, int.class, int.class, int.class});
			else if(inputType1 == InputType.VECTOR_SPARSE && inputType2 == InputType.SCALAR)
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double[].class, double.class, int[].class, int.class, int.class, int.class});
			else if(inputType1 == InputType.SCALAR && inputType2 == InputType.VECTOR_SPARSE)
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double.class, double[].class, int[].class, int.class, int.class, int.class});
			else // if( type1==InputType.VECTOR_SPARSE && type2==InputType.VECTOR_DENSE )
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double[].class, double[].class, int[].class, int.class, int.class, int.class,
						int.class});

			long nanos;
			if(inputType1 == InputType.SCALAR && inputType2 == InputType.VECTOR_DENSE) {
				nanos = System.nanoTime();
				for(int i = 0; i < m; i++) {
					me.invoke(null, inA.max(), inB.getDenseBlockValues(), i * n, n);
				}
				return System.nanoTime() - nanos;
			}else if(inputType1 == InputType.VECTOR_DENSE && inputType2 == InputType.SCALAR) {
				nanos = System.nanoTime();
				for(int i = 0; i < m; i++) {
					me.invoke(null, inA.getDenseBlockValues(), inB.max(), i * n, n);
				}
				return System.nanoTime() - nanos;
			}else if(inputType1 == InputType.VECTOR_DENSE && inputType2 == InputType.VECTOR_DENSE) {
				nanos = System.nanoTime();
				for(int i = 0; i < m; i++) {
					me.invoke(null, inA.getDenseBlockValues(), inB.getDenseBlockValues(), i * n,
						i * n, n);
				}
				return System.nanoTime() - nanos;
			}else if(inputType1 == InputType.VECTOR_SPARSE && inputType2 == InputType.SCALAR) {
				nanos = System.nanoTime();
				for(int i = 0; i < m; i++) {
					me.invoke(null, inA.getSparseBlock().values(i), inB.max(),
						inA.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i), n);
				}
				return System.nanoTime() - nanos;
			}else if(inputType1 == InputType.SCALAR && inputType2 == InputType.VECTOR_SPARSE) {
				nanos = System.nanoTime();
				for(int i = 0; i < m; i++) {
					me.invoke(null, inA.max(), inB.getSparseBlock().values(i),
						inB.getSparseBlock().indexes(i), inB.getSparseBlock().pos(i), inB.getSparseBlock().size(i), n);
				}
				return System.nanoTime() - nanos;
			}else if(inputType1 == InputType.VECTOR_SPARSE && inputType2 == InputType.VECTOR_DENSE) {
				nanos = System.nanoTime();
				for(int i = 0; i < m; i++) {
					me.invoke(null, inA.getSparseBlock().values(i), inB.getDenseBlockValues(),
						inA.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), i * n,
						inA.getSparseBlock().size(i), n);
				}
				return System.nanoTime() - nanos;
			}
			return Long.MAX_VALUE;
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}

	public void timingUtilSparsePrimitivesTest(BinType binType, InputType inputType1, InputType inputType2) {
		try {
			double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 1264);
			double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 1265);

			String meName = "vect"+ StringUtils.camelize(binType.name().split("_")[1])+"Write";
			final Method me;
			if( inputType1==InputType.VECTOR_SPARSE && inputType2==InputType.SCALAR )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{int.class, double[].class, double.class, int[].class, int.class, int.class});
			else if( inputType1==InputType.SCALAR && inputType2==InputType.VECTOR_SPARSE )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{int.class, double.class, double[].class, int[].class, int.class, int.class});
			else //if( type1==InputType.VECTOR_SPARSE && type2==InputType.VECTOR_SPARSE )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{int.class, double[].class, double[].class, int[].class, int[].class, int.class, int.class, int.class, int.class});

			System.out.println("Sparse calculation via TimingUtils:");
			double[] times;
			if(inputType1 == InputType.VECTOR_SPARSE && inputType2 == InputType.SCALAR) {
				times = TimingUtils.time(() -> {
					try {
						for(int i = 0; i < m; i++) {
							me.invoke(null, n, inA.getSparseBlock().values(i), inB.max(),
								inA.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i));
						}
					} catch(Exception ex) {
						throw new RuntimeException(ex);
					}
				}, repetitions);
				System.out.println("Sparse calculation: " + TimingUtils.stats(times));
			}else if(inputType1 == InputType.SCALAR && inputType2 == InputType.VECTOR_SPARSE) {
				times = TimingUtils.time(() -> {
					try {
						for(int i = 0; i < m; i++) {
							me.invoke(null, n, inA.max(), inB.getSparseBlock().values(i),
								inB.getSparseBlock().indexes(i), inB.getSparseBlock().pos(i), inB.getSparseBlock().size(i));
						}
					} catch(Exception ex) {
						throw new RuntimeException(ex);
					}
				}, repetitions);
				System.out.println("Sparse calculation: " + TimingUtils.stats(times));
			}else if(inputType1 == InputType.VECTOR_SPARSE && inputType2 == InputType.VECTOR_SPARSE) {
				times = TimingUtils.time(() -> {
					try {
						for(int i = 0; i < m; i++) {
							me.invoke(null, n, inA.getSparseBlock().values(i),
								inB.getSparseBlock().values(i), inA.getSparseBlock().indexes(i),
								inB.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inB.getSparseBlock().pos(i),
								inA.getSparseBlock().size(i), inB.getSparseBlock().size(i));
						}
					} catch(Exception ex) {
						throw new RuntimeException(ex);
					}
				}, repetitions);
				System.out.println("Sparse calculation: " + TimingUtils.stats(times));
			}
		}catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}

	private void timingUtilDensePrimitivesTest(BinType binType, InputType inputType1, InputType inputType2) {
		try {
			//generate input data (scalar later derived if needed)
			double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 3);
			double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 7);

			//get vector primitive via reflection
			String meName = "vect" + StringUtils.camelize(binType.name().split("_")[1]) + "Write";
			final Method me;
			if(inputType1 == InputType.SCALAR && inputType2 == InputType.VECTOR_DENSE)
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double.class, double[].class, int.class, int.class});
			else if(inputType1 == InputType.VECTOR_DENSE && inputType2 == InputType.SCALAR)
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double[].class, double.class, int.class, int.class});
			else if(inputType1 == InputType.VECTOR_DENSE && inputType2 == InputType.VECTOR_DENSE)
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double[].class, double[].class, int.class, int.class, int.class});
			else if(inputType1 == InputType.VECTOR_SPARSE && inputType2 == InputType.SCALAR)
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double[].class, double.class, int[].class, int.class, int.class, int.class});
			else if(inputType1 == InputType.SCALAR && inputType2 == InputType.VECTOR_SPARSE)
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double.class, double[].class, int[].class, int.class, int.class, int.class});
			else // if( type1==InputType.VECTOR_SPARSE && type2==InputType.VECTOR_DENSE )
				me = LibSpoofPrimitives.class.getMethod(meName,
					new Class[] {double[].class, double[].class, int[].class, int.class, int.class, int.class,
						int.class});

			System.out.println("Dense calculation via TimingUtils:");
			double[] times;
			if(inputType1 == InputType.SCALAR && inputType2 == InputType.VECTOR_DENSE) {
				times = TimingUtils.time(() -> {
					try {
						for(int i = 0; i < m; i++) {
							me.invoke(null, inA.max(), inB.getDenseBlockValues(), i * n, n);
						}
					} catch(Exception ex) {
						throw new RuntimeException(ex);
					}
				}, repetitions);
				System.out.println("Dense calculation: " + TimingUtils.stats(times));
			}else if(inputType1 == InputType.VECTOR_DENSE && inputType2 == InputType.SCALAR) {
				times = TimingUtils.time(() -> {
					try {
						for(int i = 0; i < m; i++) {
							me.invoke(null, inA.getDenseBlockValues(), inB.max(), i * n, n);
						}
					} catch(Exception ex) {
						throw new RuntimeException(ex);
					}
				}, repetitions);
				System.out.println("Dense calculation: " + TimingUtils.stats(times));
			}else if(inputType1 == InputType.VECTOR_DENSE && inputType2 == InputType.VECTOR_DENSE) {
				times = TimingUtils.time(() -> {
					try {
						for(int i = 0; i < m; i++) {
							me.invoke(null, inA.getDenseBlockValues(), inB.getDenseBlockValues(), i * n,
								i * n, n);
						}
					} catch(Exception ex) {
						throw new RuntimeException(ex);
					}
				}, repetitions);
				System.out.println("Dense calculation: " + TimingUtils.stats(times));
			}else if(inputType1 == InputType.VECTOR_SPARSE && inputType2 == InputType.SCALAR) {
				times = TimingUtils.time(() -> {
					try {
						for(int i = 0; i < m; i++) {
							me.invoke(null, inA.getSparseBlock().values(i), inB.max(),
								inA.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i), n);
						}
					} catch(Exception ex) {
						throw new RuntimeException(ex);
					}
				}, repetitions);
				System.out.println("Dense calculation: " + TimingUtils.stats(times));
			}else if(inputType1 == InputType.SCALAR && inputType2 == InputType.VECTOR_SPARSE) {
				times = TimingUtils.time(() -> {
					try {
						for(int i = 0; i < m; i++) {
							me.invoke(null, inA.max(), inB.getSparseBlock().values(i),
								inB.getSparseBlock().indexes(i), inB.getSparseBlock().pos(i), inB.getSparseBlock().size(i), n);
						}
					} catch(Exception ex) {
						throw new RuntimeException(ex);
					}
				}, repetitions);
				System.out.println("Dense calculation: " + TimingUtils.stats(times));
			}else if(inputType1 == InputType.VECTOR_SPARSE && inputType2 == InputType.VECTOR_DENSE) {
				times = TimingUtils.time(() -> {
					try {
						for(int i = 0; i < m; i++) {
							me.invoke(null, inA.getSparseBlock().values(i), inB.getDenseBlockValues(),
								inA.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), i * n,
								inA.getSparseBlock().size(i), n);
						}
					} catch(Exception ex) {
						throw new RuntimeException(ex);
					}
				}, repetitions);
				System.out.println("Dense calculation: " + TimingUtils.stats(times));
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}

	public static void main(String[] args) {
		new SparseRowPerfTest().testSparseBinaryPrimitivesPerf();
	}
}
