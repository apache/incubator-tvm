..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _tvm-runtime-vulkan:

Vulkan Runtime
==============

TVM supports using Vulkan compute shaders to execute queries.  Each
computational kernel is compiled into a SPIR-V shader, which can then
be called using the TVM interface.

.. _tvm-runtime-vulkan-features:

Vulkan Features, Limits
-----------------------

.. _Required Limits: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#limits-minmax

Since different Vulkan implementations may enable different optional
features or have different physical limits, the code generation must
know which features are available to use.  These correspond to
specific Vulkan capabilities/limits as in
:ref:`Vulkan Capabilities Table <tvm-table-vulkan-capabilities>`.
If unspecified, TVM assumes that a capability is not available, or
that a limit is the minimum guaranteed by the Vulkan spec in the
`Required Limits`_ section.

These parameters can be either explicitly specific when defining a
:ref:`Target <tvm-target-specific-target>`, or can be queried from a
device.  To query from a device, the special parameter
``-from_device=N`` can be used to query all vulkan device parameters
from device id ``N``.  Any additional parameters explicitly specified
will override the parameters queried from the device.

.. _VkSubgroupFeatureFlagBits: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkSubgroupFeatureFlagBits.html

.. list-table:: Vulkan Capabilities
   :name: tvm-runtime-table-vulkan-capabilities
   :header-rows: 1

   * - Target Parameter
     - Required Vulkan Version/Extension
     - Parameter Queried
     - Default Value

   * - ``supported_subgroup_operations``
     - Vulkan 1.1+
     - ``VkPhysicalDeviceSubgroupProperties::supportedOperations``
     - 0 (interpreted as `VkSubgroupFeatureFlagBits`_)

   * - ``max_push_constants_size``
     -
     - ``VkPhysicalDeviceLimits::maxPushConstantsSize``
     - 128 bytes

   * - ``max_uniform_buffer_range``
     -
     - ``VkPhysicalDeviceLimits::maxUniformBufferRange``
     - 16384 bytes


   * - ``max_storage_buffer_range``
     -
     - ``VkPhysicalDeviceLimits::maxStorageBufferRange``
     - 2\ :sup:`27`\ bytes


   * - ``max_per_stage_descriptor_storage_buffer``
     -
     - ``VkPhysicalDeviceLimits::maxPerStageDescriptorStorageBuffers``
     - 4


   * - ``supports_storage_buffer_storage_class``
     - VK_KHR_storage_buffer_storage_class
     -
     - false


   * - ``supports_storage_buffer_8bit_access``
     - VK_KHR_8bit_storage
     - ``VkPhysicalDevice8BitStorageFeaturesKHR::storageBuffer8BitAccess``
     - false


   * - ``supports_storage_buffer_16bit_access``
     - VK_KHR_16bit_storage
     - ``VkPhysicalDevice16BitStorageFeaturesKHR::storageBuffer16BitAccess``
     - false


   * - ``supports_float16``
     - VK_KHR_shader_float16_int8
     - ``VkPhysicalDeviceShaderFloat16Int8FeaturesKHR::shaderFloat16``
     - false


   * - ``supports_float64``
     -
     - ``VkPhysicalDeviceFeatures::shaderFloat64``
     - false


   * - ``supports_int8``
     - VK_KHR_shader_float16_int8
     - ``VkPhysicalDeviceShaderFloat16Int8FeaturesKHR::shaderInt8``
     - false


   * - ``supports_int16``
     -
     - ``VkPhysicalDeviceFeatures::shaderInt16``
     - false


   * - ``supports_int64``
     -
     - ``VkPhysicalDeviceFeatures::shaderInt64``
     - false



As of May 2021, not all Vulkan implementations are supported.  For
example, support for 64-bit integers is required.  If a Vulkan target
is not supported, an error message should be issued during SPIR-V code
generation.  Efforts are also underway to remove these requirements
and support additional Vulkan implementations.


.. _tvm-runtime-vulkan-spirv-capabilities:

SPIR-V Capabilities
-------------------

Some of the device-specific capabilities also correspond to SPIR-V
capabilities or extensions that must be declared in the shader, or a
minimum SPIR-V version required in order to use a feature.  The
TVM-generated shaders will declare the minimum set of
extensions/capabilities and the minimum allowed version of SPIR-V
that are needed to execute the compiled graph.

If the shader generation requires a capability or extension that is
not enabled in the ``Target``, an exception will be raised.


.. list-table:: Vulkan Capabilities
   :name: tvm-table-vulkan-capabilities
   :header-rows: 1

   * - Target Parameter
     - Required SPIR-V Version/Extension
     - Declared Capability

   * - ``supported_subgroup_operations``
     - SPIR-V 1.3+
     - Varies, see `VkSubgroupFeatureFlagBits`_

   * - ``supports_storage_buffer_storage_class``
     - SPV_KHR_storage_buffer_storage_class
     -

   * - ``supports_storage_buffer_8bit_access``
     - SPV_KHR_8bit_storage
     - StorageBuffer8BitAccess

   * - ``supports_storage_buffer_16bit_access``
     - SPV_KHR_16bit_storage
     - StorageBuffer16BitAccess

   * - ``supports_float16``
     -
     - Float16


   * - ``supports_float64``
     -
     - Float64


   * - ``supports_int8``
     -
     - Int8


   * - ``supports_int16``
     -
     - Int16


   * - ``supports_int64``
     -
     - Int64


Vulkan-Specific Environment Variables
-------------------------------------

Both the SPIR-V code generation and the Vulkan runtime have
environment variables that can modify some of the runtime behavior.
These are intended for debugging purposes, both to more easily test
specific code paths, and to output more information as needed.  All
boolean flags are true if the environment variable is set to a
non-zero integer.  An unset variable, the integer zero, or an empty
string are all false boolean flags.

.. _VK_KHR_push_descriptor: https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_push_descriptor.html

.. _VK_KHR_descriptor_update_template: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_descriptor_update_template.html

.. _VK_KHR_dedicated_allocation: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_dedicated_allocation.html

.. _VkMemoryDedicatedRequirements: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkMemoryDedicatedRequirements.html

.. _Vulkan validation layers: https://github.com/KhronosGroup/Vulkan-LoaderAndValidationLayers/blob/master/layers/README.md

.. _spvValidate: https://github.com/KhronosGroup/SPIRV-Tools#validator


* ``TVM_VULKAN_DISABLE_PUSH_DESCRIPTOR`` - A boolean flag.  If true,
  TVM will explicitly allocate descriptors, and will not use the
  `VK_KHR_push_descriptor`_ or `VK_KHR_descriptor_update_template`_
  extensions.  If false, TVM will decide whether to use these
  extensions based on their availability.

* ``TVM_VULKAN_DISABLE_DEDICATED_ALLOCATION`` - A boolean flag.  If
  true, TVM will not mark memory allocations as being dedicated
  allocations, and will not use the `VK_KHR_dedicated_allocation`_
  extension.  If false, TVM will decide whether memory allocations
  should be marked as dedicated based on the
  `VkMemoryDedicatedRequirements`_ for that buffer.

* ``TVM_VULKAN_ENABLE_VALIDATION_LAYERS`` - A boolean flag.  If true,
  TVM will enable `Vulkan validation layers`_ that the device
  supports.  If false, no validation layers are enabled.

* ``TVM_TIRX_VULKAN_DISABLE_SHADER_VALIDATION`` - A boolean flag.  If true,
  the SPIR-V shader validation done with `spvValidate`_ is skipped.
  If false (default), all SPIR-V shaders generated by TVM are
  validated with `spvValidate`_.

* ``TVM_TIRX_VULKAN_DEBUG_SHADER_SAVEPATH`` - A path to a directory.  If
  set to a non-empty string, the Vulkan codegen will save TIR, binary
  SPIR-V, and disassembled SPIR-V shaders to this directory, to be
  used for debugging purposes.
