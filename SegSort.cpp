#include <utility>
#include <thread>
#include <chrono>
using namespace std::chrono;

#include "SegSort.h"
#include "ComputeUtil.h"

void SegmentedSort::InitPartition(const wgpu::Device& device, const wgpu::Buffer& inputBuffer) {
  auto bgl = utils::MakeBindGroupLayout(
    device, "PartitionLayout", {
        { 0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage }, 
        { 1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform }, 
        { 2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage }, 
        { 3, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage },
        { 4, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 5, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 6, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 7, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
  });

  partitionPipeline = ComputeUtil::CreatePipeline(device, bgl,
    #include "segsort_tuple/seg_partition.wgsl"
    , "Sort::partitionPipeline"
  );

  partitionBindGroups[0] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBuffer },         
          { 1, paramBuffer, 0, sizeof(Param) },
          { 2, mergeRangesBuffer },
          { 3, compressedRangesBuffer },
          { 4, passCountBuffer },
          { 5, opCounterBuffer },
          { 6, mergeListBuffer },
          { 7, copyListBuffer },
    });
  
  partitionBindGroups[1] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBufferCopy },         
          { 1, paramBuffer, 0, sizeof(Param) },
          { 2, mergeRangesBuffer },
          { 3, compressedRangesBuffer },
          { 4, passCountBuffer },
          { 5, opCounterBuffer },
          { 6, mergeListBuffer },
          { 7, copyListBuffer },
    });
}

void SegmentedSort::InitClear(const wgpu::Device& device) {
  auto bgl = utils::MakeBindGroupLayout(
    device, "ClearLayout", {
        { 0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform },
        { 1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage }, 
  });

  clearPipeline = ComputeUtil::CreatePipeline(device, bgl,
    #include "segsort_tuple/seg_clear.wgsl"
    , "Sort::clearPipeline"
  );

  clearBindGroup = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, paramBuffer, 0, sizeof(Param) },
          { 1, opCounterBuffer },
          { 2, passCountBuffer },
    });
}

void SegmentedSort::InitCopy(const wgpu::Device& device, const wgpu::Buffer& inputBuffer) {
  auto bgl = utils::MakeBindGroupLayout(
    device, "CopyLayout", {
        { 0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage },
        { 1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage }, 
        { 3, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform }, 
  });

  copyPipeline = ComputeUtil::CreatePipeline(device, bgl,
    #include "segsort_tuple/seg_copy.wgsl"
    , "Sort::copyPipeline"
  );

  copyBindGroups[0] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBuffer },
          { 1, inputBufferCopy },
          { 2, copyListBuffer },
          { 3, paramBuffer, 0, sizeof(Param) },
    });
  copyBindGroups[1] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBufferCopy },
          { 1, inputBuffer },
          { 2, copyListBuffer },
          { 3, paramBuffer, 0, sizeof(Param) },
    });
} 

void SegmentedSort::InitBlock(
  const wgpu::Device& device, 
  const wgpu::Buffer& inputBuffer, 
  const wgpu::Buffer& segmentsBuffer
) {

  {
    auto bgl0 = utils::MakeBindGroupLayout(
    device, "BlockLayout0", {
        { 0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform }, 
        { 2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage },
        { 3, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage },
        { 4, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
  });

  blockPipeline[0] = ComputeUtil::CreatePipeline(device, bgl0,
    #include "segsort_tuple/seg_block_0.wgsl"
    , "Sort::blockPipeline0"
  );


  blockBindGroups[0] = utils::MakeBindGroup(
    device, bgl0,
        {
          { 0, inputBuffer },
          { 1, paramBuffer, 0, sizeof(Param) },
          { 2, segmentsBuffer },
          { 3, partitionBuffer },
          { 4, compressedRangesBuffer },
    });
  }

  {
    auto bgl = utils::MakeBindGroupLayout(
    device, "BlockLayout1", {
        { 0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage },
        { 1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform }, 
        { 3, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage },
        { 4, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage },
        { 5, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
  });

  blockPipeline[1] = ComputeUtil::CreatePipeline(device, bgl,
    #include "segsort_tuple/seg_block.wgsl"
    , "Sort::blockPipeline1"
  );

  blockBindGroups[1] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBuffer },
          { 1, inputBufferCopy },
          { 2, paramBuffer, 0, sizeof(Param) },
          { 3, segmentsBuffer },
          { 4, partitionBuffer },
          { 5, compressedRangesBuffer },
    });
  }
} 

void SegmentedSort::InitBinarySearch(const wgpu::Device& device, const wgpu::Buffer& segmentsBuffer) {
  auto bgl = utils::MakeBindGroupLayout(
    device, "BinarySearchLayout", {
        { 0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage },
        { 1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform }, 
  });

  binarySearchPipeline = ComputeUtil::CreatePipeline(device, bgl,
    #include "segsort_tuple/binary_search.wgsl"
    , "Sort::binarySearchPipeline"
  );

  binarySearchBindGroup = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, segmentsBuffer },
          { 1, partitionBuffer },
          { 2, paramBuffer, 0, sizeof(Param) },
    });
} 

void SegmentedSort::InitMerge(const wgpu::Device& device, const wgpu::Buffer& inputBuffer) {
  auto bgl = utils::MakeBindGroupLayout(
    device, "MergeLayout", {
        { 0, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage },
        { 1, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Storage },
        { 2, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::Uniform }, 
        { 3, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage },
        { 4, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage }, 
        { 5, wgpu::ShaderStage::Compute, wgpu::BufferBindingType::ReadOnlyStorage }, 
  });

  mergePipeline = ComputeUtil::CreatePipeline(device, bgl,
    #include "segsort_tuple/seg_merge.wgsl"
    , "Sort::mergePipeline"
  );
 
  mergeBindGroups[0] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBuffer },
          { 1, inputBufferCopy },
          { 2, paramBuffer, 0, sizeof(Param) },
          { 3, mergeListBuffer },
          { 4, compressedRangesBuffer },
          { 5, passCountBuffer },

    });

  mergeBindGroups[1] = utils::MakeBindGroup(
    device, bgl,
        {
          { 0, inputBufferCopy },
          { 1, inputBuffer },
          { 2, paramBuffer, 0, sizeof(Param) },
          { 3, mergeListBuffer },
          { 4, compressedRangesBuffer },
          { 5, passCountBuffer },
    });
} 

void SegmentedSort::Dispose() {
  inputBufferCopy.Destroy();
  partitionBuffer.Destroy();
  paramBuffer.Destroy();
  passCountBuffer.Destroy();
  compressedRangesBuffer.Destroy();
  mergeRangesBuffer.Destroy();
  mergeListBuffer.Destroy();
  copyListBuffer.Destroy();
  opCounterBuffer.Destroy();
}

void SegmentedSort::Init(
  const wgpu::Device& device,
  const wgpu::Buffer& inputBuffer, 
  uint32_t maxInputSize, 
  const wgpu::Buffer& segmentBuffer, 
  uint32_t maxSegmentSize
) {
    maxCount = maxInputSize;
    maxNumCtas = ComputeUtil::div_up(maxCount, nv);
    maxNumPasses = ComputeUtil::find_log2(maxNumCtas, true);
    maxNumSegments = maxSegmentSize; 
    maxCapacity = maxNumCtas;          
    for (int i = 0; i < maxNumPasses; i++) {
      maxCapacity += ComputeUtil::div_up(maxNumCtas, 1 << i);
    }

    InitBuffers(device);
    InitBlock(device, inputBuffer, segmentBuffer);
    InitBinarySearch(device, segmentBuffer);
    InitPartition(device, inputBuffer);
    InitMerge(device, inputBuffer);
    InitCopy(device, inputBuffer);   
    InitClear(device);
}

void SegmentedSort::InitBuffers(const wgpu::Device& device) {
    auto usage = wgpu::BufferUsage::Storage;

    compressedRangesBuffer = utils::CreateBuffer(
      device, 
      maxNumCtas * sizeof(int), 
      usage,
      "SegSort::compressedRanges"
    );
    mergeRangesBuffer = utils::CreateBuffer(
      device,
      maxCapacity * sizeof(uint32_t) * 4,
      usage,
      "SegSort::mergeRanges"
    );

    uint32_t np = std::max(maxNumPasses, 1u);
    mergeListBuffer = utils::CreateBuffer(
      device, 
      maxNumCtas * sizeof(int) * 4, 
      usage,
      "SegSort::mergeList"
    );

    if (maxNumCtas * sizeof(int) > COPY_STATUS_OFFSET*4) {
      std::cerr << "Need bigger offset for copy status buffer" << std::endl;
      exit(1);
    }
    copyListBuffer = utils::CreateBuffer(
      device, 
      maxNumCtas * sizeof(int) + COPY_STATUS_OFFSET*4 + maxNumCtas * sizeof(int),
      usage,
      "SegSort::copyList"
    );

    opCounterBuffer = utils::CreateBuffer(
      device, 
      np*sizeof(int) * 12, 
      usage | wgpu::BufferUsage::Indirect,
      "SegSort::opCounter"
    );

    inputBufferCopy = utils::CreateBuffer(
      device, 
      maxCount * sizeof(int) * 2, 
      wgpu::BufferUsage::Storage,
      "SegSort::inputBufferCopy"
    );

    paramBuffer = utils::CreateBuffer(
      device, 
      sizeof(Param), 
      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
      "SegSort::paramBuffer"
    );

    partitionBuffer = utils::CreateBuffer(
      device,
      sizeof(int) * maxNumSegments,
      wgpu::BufferUsage::Storage,
      "SegSort::partitionBuffer"
    );
    
    passCountBuffer = utils::CreateBuffer(
      device,
      sizeof(int),
      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
      "SegSort::passCountBuffer"
    );
}

void SegmentedSort::Clear(const wgpu::CommandEncoder& encoder) {
  // TODO: when fillBUffer -> fill opCounter with 1s and remove the clear pass 
  // encoder.ClearBuffer(passCountBuffer, 0, 4);
  // encoder.ClearBuffer(opCounterBuffer, 0, maxNumPasses * 12 * sizeof(uint32_t));
}

void SegmentedSort::Upload(const wgpu::Device& device, uint32_t count, uint32_t segmentCount) {
   if (count > maxCount || segmentCount > maxNumSegments) {
    std::cerr << "SegmentedSort: need to resize (" << count  << "," << maxCount << ")";
    exit(1);
  }
  uint32_t numCtas = ComputeUtil::div_up(count, nv);
  uint32_t numPasses = ComputeUtil::find_log2(numCtas, true);

  uint8_t blockBindgroupIndex = 1 & numPasses;
  int num_partitions = numCtas + 1;
  int num_partition_ctas = ComputeUtil::div_up(num_partitions, nt2 - 1);

  if (count != previousCount) {
    params.nt = nt;
    params.vt = vt;
    params.count = count;
    params.num_partitions = num_partitions;
    params.num_segments = segmentCount;
    params.nt2 = nt2;
    params.num_partition_ctas = num_partition_ctas;
    params.num_ranges = numCtas;
    params.max_num_passes = maxNumPasses;
    device.GetQueue().WriteBuffer(paramBuffer, 0, &params, sizeof(Param));
  }
}

void SegmentedSort::Sort(
  const wgpu::Device& device, 
  const wgpu::ComputePassEncoder& computePass, 
  uint32_t count, 
  uint32_t segmentCount
) {
  if (count > maxCount || segmentCount > maxNumSegments) {
    std::cerr << "SegmentedSort: need to resize (" << count  << "," << maxCount << ")";
    exit(1);
  }
  uint32_t numCtas = ComputeUtil::div_up(count, nv);
  uint32_t numPasses = ComputeUtil::find_log2(numCtas, true);

  uint8_t blockBindgroupIndex = 1 & numPasses;
  int num_partitions = numCtas + 1;
  int num_partition_ctas = ComputeUtil::div_up(num_partitions, nt2 - 1);
  uint32_t numBinarySearchDispatch = ComputeUtil::div_up(num_partitions, nv);

  computePass.SetPipeline(clearPipeline);
  computePass.SetBindGroup(0, clearBindGroup);
  computePass.DispatchWorkgroups(ComputeUtil::div_up(maxNumPasses * 24, nv));
 
  computePass.SetPipeline(binarySearchPipeline);
  computePass.SetBindGroup(0, binarySearchBindGroup);
  computePass.DispatchWorkgroups(numBinarySearchDispatch);

  computePass.SetPipeline(blockPipeline[blockBindgroupIndex]);
  computePass.SetBindGroup(0, blockBindGroups[blockBindgroupIndex]);
  computePass.DispatchWorkgroups(numCtas);
  
  uint32_t mergeBindgroupIndex = 0;
  if (1 & numPasses) {
    mergeBindgroupIndex++;
  }

  for (int pass = 0; pass < numPasses; pass++) {
    computePass.SetPipeline(partitionPipeline);
    computePass.SetBindGroup(0, partitionBindGroups[mergeBindgroupIndex % 2]);
    computePass.DispatchWorkgroups(num_partition_ctas);
    
    computePass.SetPipeline(mergePipeline);
    computePass.SetBindGroup(0, mergeBindGroups[mergeBindgroupIndex % 2]);
    computePass.DispatchWorkgroupsIndirect(opCounterBuffer, (pass * 6) * sizeof(int));

    computePass.SetPipeline(copyPipeline);
    computePass.SetBindGroup(0, copyBindGroups[mergeBindgroupIndex % 2]);
    computePass.DispatchWorkgroupsIndirect(opCounterBuffer, ((pass*2+1) * 3) * sizeof(int));
    mergeBindgroupIndex++;
  }

  previousCount = count;
}
