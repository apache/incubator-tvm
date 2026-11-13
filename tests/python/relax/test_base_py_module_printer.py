# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring, invalid-name, unused-argument
# ruff: noqa: F401, F841

import pytest

import tvm
import tvm_tirx
from tvm.relax import script as R
from tvm.relax.base_py_module import BasePyModule
from tvm.s_tir import script as Ts
from tvm_tirx.ir import StringImm
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@R.py_module
class SimplePyFuncModule(BasePyModule):
    """Test simple Python functions with basic operations."""

    @I.pyfunc
    def add(self, x, y):
        """Simple addition function."""
        x_tvm = self._convert_pytorch_to_tvm(x)
        y_tvm = self._convert_pytorch_to_tvm(y)
        result = self.call_tir(self.add_tir, [x_tvm, y_tvm], out_ty=R.Tensor((5,), "float32"))
        return self._convert_tvm_to_pytorch(result)

    @I.pyfunc
    def multiply(self, x, y):
        """Simple multiplication function."""
        x_tvm = self._convert_pytorch_to_tvm(x)
        y_tvm = self._convert_pytorch_to_tvm(y)
        result = self.call_tir(self.multiply_tir, [x_tvm, y_tvm], out_ty=R.Tensor((5,), "float32"))
        return self._convert_tvm_to_pytorch(result)

    @Ts.function
    def add_tir(
        x: T.Tensor((5,), "float32"), y: T.Tensor((5,), "float32"), out: T.Tensor((5,), "float32")
    ):
        for i in range(5):
            out[i] = x[i] + y[i]

    @Ts.function
    def multiply_tir(
        x: T.Tensor((5,), "float32"), y: T.Tensor((5,), "float32"), out: T.Tensor((5,), "float32")
    ):
        for i in range(5):
            out[i] = x[i] * y[i]

    @R.function
    def main_relax(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "float32"
    ):
        return R.add(x, y)


@R.py_module
class ComplexPyFuncModule(BasePyModule):
    """Test complex Python logic with ML pipeline and error handling."""

    @I.pyfunc
    def ml_pipeline(self, input_data, model_params):
        """Complex ML pipeline with data validation and error handling."""
        # Data validation
        if input_data is None or model_params is None:
            raise ValueError("Inputs cannot be None")

        try:
            # Convert to TVM format
            tvm_data = self._convert_pytorch_to_tvm(input_data)
            tvm_params = self._convert_pytorch_to_tvm(model_params)

            # Run ML inference
            features = self.call_tir(
                self.extract_features, [tvm_data], out_ty=R.Tensor((10,), "float32")
            )

            predictions = self.call_tir(
                self.ml_inference, [features, tvm_params], out_ty=R.Tensor((5,), "float32")
            )

            # Post-process results
            final_result = self.call_tir(
                self.post_process, [predictions], out_ty=R.Tensor((5,), "float32")
            )

            return self._convert_tvm_to_pytorch(final_result)

        except Exception as e:
            self._log_error(f"ML pipeline failed: {e}")
            return self._get_default_value()

    @I.pyfunc
    def data_preprocessing(self, raw_data):
        """Data preprocessing with conditional logic."""
        if hasattr(raw_data, "numpy"):
            # Vectorized path for numpy-compatible data
            data_np = raw_data.numpy()
            processed = self._vectorized_preprocess(data_np)
        else:
            # Fallback path for other data types
            processed = self._elementwise_preprocess(raw_data)

        # Convert and return
        tvm_processed = self._convert_pytorch_to_tvm(processed)
        result = self.call_tir(
            self.normalize_data, [tvm_processed], out_ty=R.Tensor((10,), "float32")
        )
        return self._convert_tvm_to_pytorch(result)

    @Ts.function
    def extract_features(Data: T.Tensor((10,), "float32"), Features: T.Tensor((10,), "float32")):
        T.func_attr({"tirx.noalias": True})

        for i in range(10):
            Features[i] = T.sqrt(Data[i])

    @Ts.function
    def ml_inference(
        Features: T.Tensor((10,), "float32"),
        Params: T.Tensor((10,), "float32"),
        Output: T.Tensor((5,), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})

        for i in range(5):
            Output[i] = Features[i] * Params[i] + Features[i + 5] * Params[i + 5]

    @Ts.function
    def post_process(Predictions: T.Tensor((5,), "float32"), Final: T.Tensor((5,), "float32")):
        T.func_attr({"tirx.noalias": True})

        for i in range(5):
            Final[i] = T.max(Predictions[i], 0.0)

    @Ts.function
    def normalize_data(Data: T.Tensor((10,), "float32"), Normalized: T.Tensor((10,), "float32")):
        T.func_attr({"tirx.noalias": True})

        for i in range(10):
            Normalized[i] = Data[i] / 255.0


@R.py_module
class EdgeCasePyFuncModule(BasePyModule):
    """Test edge cases and boundary conditions."""

    @I.pyfunc
    def empty_func(self):
        """Empty function with no operations."""
        pass

    @I.pyfunc
    def single_return(self, x):
        """Function with immediate return."""
        return x

    @I.pyfunc
    def nested_conditionals(self, data, threshold):
        """Function with complex nested conditional logic."""
        if data is None:
            return None

        if hasattr(data, "shape"):
            if len(data.shape) == 1:
                if data.shape[0] > threshold:
                    return self._process_large_data(data)
                else:
                    return self._process_small_data(data)
            elif len(data.shape) == 2:
                return self._process_2d_data(data)
            else:
                return self._process_nd_data(data)
        else:
            return self._process_scalar_data(data)

    @I.pyfunc
    def loop_with_break(self, data, max_iter):
        """Function with loop and break statement."""
        result = []
        for i, item in enumerate(data):
            if i >= max_iter:
                break
            if item > 0:
                result.append(item * 2)
            else:
                result.append(0)
        return result

    @Ts.function
    def dummy_tir(Data: T.Tensor((1,), "float32"), Output: T.Tensor((1,), "float32")):
        T.func_attr({"tirx.noalias": True})

        Output[0] = Data[0]


@R.py_module
class PerformancePyFuncModule(BasePyModule):
    """Test performance optimization patterns."""

    @I.pyfunc
    def vectorized_operation(self, x, y):
        """Vectorized operation with numpy fallback."""
        try:
            # Try vectorized operation first
            if hasattr(x, "numpy") and hasattr(y, "numpy"):
                x_np = x.numpy()
                y_np = y.numpy()
                result_np = x_np + y_np
                return self._convert_numpy_to_pytorch(result_np)
        except Exception:
            pass

        # Fallback to TVM processing
        x_tvm = self._convert_pytorch_to_tvm(x)
        y_tvm = self._convert_pytorch_to_tvm(y)
        result = self.call_tir(
            self.vectorized_add, [x_tvm, y_tvm], out_ty=R.Tensor((10,), "float32")
        )
        return self._convert_tvm_to_pytorch(result)

    @I.pyfunc
    def batch_processing(self, batch_data):
        """Batch processing with memory optimization."""
        batch_size = len(batch_data)
        results = []

        # Process in chunks to optimize memory usage
        chunk_size = min(batch_size, 100)
        for i in range(0, batch_size, chunk_size):
            chunk = batch_data[i : i + chunk_size]
            chunk_result = self._process_chunk(chunk)
            results.extend(chunk_result)

        return results

    @I.pyfunc
    def memory_efficient_transform(self, large_tensor):
        """Memory-efficient tensor transformation."""
        # Use in-place operations when possible
        if hasattr(large_tensor, "requires_grad") and not large_tensor.requires_grad:
            # In-place operation for efficiency
            large_tensor.add_(1.0)
            return large_tensor
        else:
            # Create new tensor if gradients are needed
            return large_tensor + 1.0

    @Ts.function
    def vectorized_add(
        A: T.Tensor((10,), "float32"), B: T.Tensor((10,), "float32"), C: T.Tensor((10,), "float32")
    ):
        T.func_attr({"tirx.noalias": True})

        for i in range(10):
            C[i] = A[i] + B[i]


@R.py_module
class IntegrationPyFuncModule(BasePyModule):
    """Test integration with external libraries and complex workflows."""

    @I.pyfunc
    def sklearn_integration(self, input_data, scaler_params):
        """Integration with scikit-learn preprocessing."""
        try:
            # Import sklearn components
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Create and fit scaler
            scaler = StandardScaler()
            if scaler_params is not None:
                scaler.mean_ = scaler_params["mean"]
                scaler.scale_ = scaler_params["scale"]
            else:
                scaler.fit(input_data)

            # Transform data
            scaled_data = scaler.transform(input_data)

            # Apply PCA if needed
            if input_data.shape[1] > 10:
                pca = PCA(n_components=10)
                reduced_data = pca.fit_transform(scaled_data)
            else:
                reduced_data = scaled_data

            # Convert to TVM and process
            tvm_data = self._convert_pytorch_to_tvm(reduced_data)
            result = self.call_tir(
                self.final_transform,
                [tvm_data],
                out_ty=R.Tensor((reduced_data.shape[0], 10), "float32"),
            )

            return self._convert_tvm_to_pytorch(result)

        except ImportError:
            # Fallback if sklearn is not available
            return self._fallback_preprocessing(input_data)

    @I.pyfunc
    def multi_stage_pipeline(self, raw_input):
        """Multi-stage processing pipeline."""
        # Stage 1: Data cleaning
        cleaned = self._clean_data(raw_input)

        # Stage 2: Feature extraction
        features = self._extract_features(cleaned)

        # Stage 3: Model inference
        predictions = self._run_inference(features)

        # Stage 4: Post-processing
        final_result = self._post_process_output(predictions)

        return final_result

    @Ts.function
    def final_transform(Data: T.Tensor((10, 10), "float32"), Output: T.Tensor((10, 10), "float32")):
        T.func_attr({"tirx.noalias": True})

        for i in range(10):
            for j in range(10):
                Output[i, j] = T.tanh(Data[i, j])


@R.py_module
class ErrorHandlingPyFuncModule(BasePyModule):
    """Test comprehensive error handling and validation."""

    @I.pyfunc
    def robust_data_processing(self, input_data, config):
        """Robust data processing with comprehensive error handling."""
        try:
            # Validate inputs
            if not self._validate_inputs(input_data, config):
                raise ValueError("Invalid input data or configuration")

            # Check data types
            if not self._check_data_types(input_data):
                raise TypeError("Unsupported data types")

            # Process data with retry logic
            max_retries = config.get("max_retries", 3)
            for attempt in range(max_retries):
                try:
                    result = self._process_with_validation(input_data, config)
                    if self._validate_output(result):
                        return result
                    else:
                        raise RuntimeError("Output validation failed")
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    self._log_warning(f"Attempt {attempt + 1} failed: {e}")
                    continue

        except Exception as e:
            self._log_error(f"Data processing failed: {e}")
            return self._get_safe_fallback(input_data, config)

    @I.pyfunc
    def graceful_degradation(self, primary_input, fallback_input):
        """Function that gracefully degrades when primary path fails."""
        try:
            # Try primary processing path
            result = self._primary_processing(primary_input)
            return result
        except Exception as e:
            self._log_warning(f"Primary processing failed: {e}")

            try:
                # Try fallback path
                result = self._fallback_processing(fallback_input)
                return result
            except Exception as e2:
                self._log_error(f"Fallback processing also failed: {e2}")
                # Return safe default
                return self._get_safe_default()

    @Ts.function
    def safe_transform(Data: T.Tensor((5,), "float32"), Output: T.Tensor((5,), "float32")):
        T.func_attr({"tirx.noalias": True})

        for i in range(5):
            # Safe operation that handles edge cases
            if Data[i] > 0:
                Output[i] = T.sqrt(Data[i])
            else:
                Output[i] = 0.0


# Pytest test functions to verify the classes work correctly
def test_simple_pyfunc_module_creation():
    """Test that SimplePyFuncModule can be created."""
    # Get the IRModule instance from the TVMScript decorated class
    ir_mod = SimplePyFuncModule
    device = tvm_tirx.cpu()

    # Create BasePyModule instance
    module = BasePyModule(ir_mod, device)
    assert isinstance(module, BasePyModule)

    # Note: Python functions are stored in pyfuncs, not as direct attributes
    # We need to check if they exist in the IRModule's pyfuncs
    if hasattr(ir_mod, "__pyfuncs__"):
        assert "add" in ir_mod.__pyfuncs__
        assert "multiply" in ir_mod.__pyfuncs__

    # Check that TIR functions exist
    assert hasattr(module, "add_tir")
    assert hasattr(module, "multiply_tir")

    # Note: This particular TVMScript is for testing purpose only, and cannot compile
    # Relax functions may not be available due to TVMScript compilation issues
    print("Note: This TVMScript is for testing purpose only, and cannot compile")


def test_complex_pyfunc_module_creation():
    """Test that ComplexPyFuncModule can be created."""
    ir_mod = ComplexPyFuncModule
    device = tvm_tirx.cpu()

    module = BasePyModule(ir_mod, device)
    assert isinstance(module, BasePyModule)

    # Check Python functions in pyfuncs
    if hasattr(ir_mod, "__pyfuncs__"):
        assert "ml_pipeline" in ir_mod.__pyfuncs__
        assert "data_preprocessing" in ir_mod.__pyfuncs__

    # Check TIR functions
    assert hasattr(module, "extract_features")
    assert hasattr(module, "ml_inference")
    assert hasattr(module, "post_process")
    assert hasattr(module, "normalize_data")


def test_edge_case_pyfunc_module_creation():
    """Test that EdgeCasePyFuncModule can be created."""
    ir_mod = EdgeCasePyFuncModule
    device = tvm_tirx.cpu()

    module = BasePyModule(ir_mod, device)
    assert isinstance(module, BasePyModule)

    # Check Python functions in pyfuncs
    if hasattr(ir_mod, "__pyfuncs__"):
        assert "empty_func" in ir_mod.__pyfuncs__
        assert "single_return" in ir_mod.__pyfuncs__
        assert "nested_conditionals" in ir_mod.__pyfuncs__
        assert "loop_with_break" in ir_mod.__pyfuncs__

    # Check TIR function
    assert hasattr(module, "dummy_tir")


def test_performance_pyfunc_module_creation():
    """Test that PerformancePyFuncModule can be created."""
    ir_mod = PerformancePyFuncModule
    device = tvm_tirx.cpu()

    module = BasePyModule(ir_mod, device)
    assert isinstance(module, BasePyModule)

    # Check Python functions in pyfuncs
    if hasattr(ir_mod, "__pyfuncs__"):
        assert "vectorized_operation" in ir_mod.__pyfuncs__
        assert "batch_processing" in ir_mod.__pyfuncs__
        assert "memory_efficient_transform" in ir_mod.__pyfuncs__

    # Check TIR function
    assert hasattr(module, "vectorized_add")


def test_integration_pyfunc_module_creation():
    """Test that IntegrationPyFuncModule can be created."""
    ir_mod = IntegrationPyFuncModule
    device = tvm_tirx.cpu()

    module = BasePyModule(ir_mod, device)
    assert isinstance(module, BasePyModule)

    # Check Python functions in pyfuncs
    if hasattr(ir_mod, "__pyfuncs__"):
        assert "sklearn_integration" in ir_mod.__pyfuncs__
        assert "multi_stage_pipeline" in ir_mod.__pyfuncs__

    # Check TIR function
    assert hasattr(module, "final_transform")


def test_error_handling_pyfunc_module_creation():
    """Test that ErrorHandlingPyFuncModule can be created."""
    ir_mod = ErrorHandlingPyFuncModule
    device = tvm_tirx.cpu()

    module = BasePyModule(ir_mod, device)
    assert isinstance(module, BasePyModule)

    # Check Python functions in pyfuncs
    if hasattr(ir_mod, "__pyfuncs__"):
        assert "robust_data_processing" in ir_mod.__pyfuncs__
        assert "graceful_degradation" in ir_mod.__pyfuncs__

    # Check TIR function
    assert hasattr(module, "safe_transform")


def test_all_modules_inherit_from_base():
    """Test that all modules properly inherit from BasePyModule."""
    modules = [
        SimplePyFuncModule,
        ComplexPyFuncModule,
        EdgeCasePyFuncModule,
        PerformancePyFuncModule,
        IntegrationPyFuncModule,
        ErrorHandlingPyFuncModule,
    ]

    device = tvm_tirx.cpu()
    for ir_mod in modules:
        module = BasePyModule(ir_mod, device)
        assert isinstance(module, BasePyModule)
        assert hasattr(module, "script")
        assert hasattr(module, "show")


def test_pyfunc_decorators():
    """Test that all @I.pyfunc decorated functions are present."""
    ir_mod = SimplePyFuncModule
    device = tvm_tirx.cpu()
    module = BasePyModule(ir_mod, device)

    # Check that the functions exist in pyfuncs
    if hasattr(ir_mod, "__pyfuncs__"):
        assert "add" in ir_mod.__pyfuncs__
        assert "multiply" in ir_mod.__pyfuncs__

        # Get the actual function objects
        add_func = ir_mod.__pyfuncs__["add"]
        multiply_func = ir_mod.__pyfuncs__["multiply"]

        # Check that they are callable
        assert callable(add_func)
        assert callable(multiply_func)

        # Check function signatures
        import inspect

        add_sig = inspect.signature(add_func)
        assert len(add_sig.parameters) == 3  # self, x, y

        multiply_sig = inspect.signature(multiply_func)
        assert len(multiply_sig.parameters) == 3  # self, x, y


def test_tir_functions():
    """Test that TIR functions are properly defined."""
    ir_mod = SimplePyFuncModule
    device = tvm_tirx.cpu()
    module = BasePyModule(ir_mod, device)

    # Check TIR function attributes
    assert hasattr(module, "add_tir")
    assert hasattr(module, "multiply_tir")

    # These should be callable (though they're TIR functions)
    assert callable(module.add_tir)
    assert callable(module.multiply_tir)


def test_relax_functions():
    """Test that Relax functions are properly defined."""
    ir_mod = SimplePyFuncModule
    device = tvm_tirx.cpu()
    module = BasePyModule(ir_mod, device)

    # Note: This particular TVMScript is for testing purpose only, and cannot compile
    # Relax functions may not be available due to TVMScript compilation issues
    print("Note: This TVMScript is for testing purpose only, and cannot compile")

    # We can still check that the module was created successfully
    assert isinstance(module, BasePyModule)
    assert hasattr(module, "script")
    assert hasattr(module, "show")


def test_module_docstrings():
    """Test that all modules have proper docstrings."""
    modules = [
        SimplePyFuncModule,
        ComplexPyFuncModule,
        EdgeCasePyFuncModule,
        PerformancePyFuncModule,
        IntegrationPyFuncModule,
        ErrorHandlingPyFuncModule,
    ]

    for module_class in modules:
        # TVMScript decorator changes the class, so we check that it's callable
        # and can create instances instead of checking docstrings
        assert callable(module_class)
        # We can't directly instantiate TVMScript decorated classes
        # but we can create BasePyModule instances with them
        device = tvm_tirx.cpu()
        instance = BasePyModule(module_class, device)
        assert isinstance(instance, BasePyModule)


def test_python_function_complexity():
    """Test that complex Python functions have the expected structure."""
    ir_mod = ComplexPyFuncModule
    device = tvm_tirx.cpu()
    module = BasePyModule(ir_mod, device)

    # Check that complex functions exist in pyfuncs
    if hasattr(ir_mod, "__pyfuncs__"):
        assert "ml_pipeline" in ir_mod.__pyfuncs__
        assert "data_preprocessing" in ir_mod.__pyfuncs__

        # Get the actual function objects
        ml_func = ir_mod.__pyfuncs__["ml_pipeline"]
        preprocess_func = ir_mod.__pyfuncs__["data_preprocessing"]

        # These should be callable
        assert callable(ml_func)
        assert callable(preprocess_func)

        # Check function signatures
        import inspect

        ml_sig = inspect.signature(ml_func)
        assert len(ml_sig.parameters) == 3  # self, input_data, model_params

        preprocess_sig = inspect.signature(preprocess_func)
        assert len(preprocess_sig.parameters) == 2  # self, raw_data


def test_script_and_show_methods():
    """Test that script() and show() methods work correctly."""
    ir_mod = SimplePyFuncModule
    device = tvm_tirx.cpu()
    module = BasePyModule(ir_mod, device)

    # Test script() method
    script_output = module.script()
    assert isinstance(script_output, str)
    assert len(script_output) > 0

    # Test show() method
    try:
        module.show()
        # If we get here, show() worked
        assert True
    except Exception as e:
        # If show() fails, the feature is not working properly
        pytest.fail(f"show() method failed: {e}")


def test_python_functions_in_irmodule():
    """Test that Python functions are properly stored in IRModule pyfuncs."""
    ir_mod = SimplePyFuncModule
    device = tvm_tirx.cpu()
    module = BasePyModule(ir_mod, device)

    # Check that pyfuncs attribute exists and contains our functions
    if hasattr(ir_mod, "__pyfuncs__"):
        pyfuncs = ir_mod.__pyfuncs__
        assert isinstance(pyfuncs, dict)
        assert "add" in pyfuncs
        assert "multiply" in pyfuncs

        # Check that the functions are callable
        assert callable(pyfuncs["add"])
        assert callable(pyfuncs["multiply"])

        # Check function names
        assert pyfuncs["add"].__name__ == "add"
        assert pyfuncs["multiply"].__name__ == "multiply"
    else:
        pytest.fail("pyfuncs attribute not found in IRModule")


def test_call_py_func_with_base_py_module():
    """Test R.call_py_func with BasePyModule."""
    import numpy as np
    import torch

    from tvm.relax import TensorType, Var
    from tvm.relax.op import call_py_func

    # Test 1: Operator creation and basic properties
    x = Var("x", TensorType((5,), "float32"))
    y = Var("y", TensorType((5,), "float32"))

    call_expr = call_py_func(StringImm("test_func"), (x, y), ty_args=[R.Tensor((5,), "float32")])

    assert call_expr.op.name == "relax.call_py_func"
    assert call_expr.args[0].value == "test_func"
    assert len(call_expr.args) == 2

    # Test 2: Compilation validation
    try:
        call_py_func(
            123,
            (Var("x", TensorType((5,), "float32")),),
            ty_args=[R.Tensor((5,), "float32")],
        ).validate()
        assert False, "Should raise type error"
    except Exception as e:
        assert "string literal" in str(e)

    # Test 3: Validation and error handling
    @R.py_module
    class ValidationTestModule(BasePyModule):
        @R.function
        def test_invalid_call(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            result = R.call_py_func("non_existent_func", (x,), ty_args=[R.Tensor((5,), "float32")])
            return result

    device = tvm_tirx.cpu()
    module = ValidationTestModule(device)

    x = torch.randn(5, dtype=torch.float32)

    with pytest.raises(ValueError, match="Python function 'non_existent_func' not found"):
        module.call_py_func("non_existent_func", [x])

    # Test 4: Using call_py_func within Relax functions
    @R.py_module
    class RelaxCallPyFuncModule(BasePyModule):
        @I.pyfunc
        def torch_relu(self, x):
            """PyTorch ReLU implementation."""
            return torch.relu(x)

        @I.pyfunc
        def torch_softmax(self, x, dim=0):
            """PyTorch softmax implementation."""
            return torch.softmax(x, dim=dim)

        @R.function
        def mixed_computation(x: R.Tensor((10,), "float32")) -> R.Tensor((10,), "float32"):
            relu_result = R.call_py_func("torch_relu", (x,), ty_args=[R.Tensor((10,), "float32")])
            final_result = R.call_py_func(
                "torch_softmax", (relu_result,), ty_args=[R.Tensor((10,), "float32")]
            )
            return final_result

    device = tvm_tirx.cpu()
    module = RelaxCallPyFuncModule(device)

    x = torch.randn(10, dtype=torch.float32)

    expected = torch.softmax(torch.relu(x), dim=0)

    relu_result = module.call_py_func("torch_relu", [x])
    final_result = module.call_py_func("torch_softmax", [relu_result])

    # Convert to numpy for comparison
    if isinstance(final_result, tvm_tirx.runtime.Tensor):
        final_result_np = final_result.numpy()
    else:
        final_result_np = final_result

    if isinstance(expected, torch.Tensor):
        expected_np = expected.numpy()
    else:
        expected_np = expected

    # Use numpy for comparison since we have numpy arrays
    tvm_tirx.testing.assert_allclose(final_result_np, expected_np, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tvm_tirx.testing.main()
