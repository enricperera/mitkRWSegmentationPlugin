/*=========================================================================
 *
 * Copyright Universitat Pompeu Fabra, Department of Information and
 * Comunication Technologies.
 *  
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef itkCudaRWSegmentationFilter_hxx
#define itkCudaRWSegmentationFilter_hxx

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#define CLEANUP(s)     \
  do                   \
  {                    \
    printf("%s\n", s); \
    fflush(stdout);    \
  } while (0)

namespace itk
{
template <typename TInputImage, typename TOutputImage>
void CudaRWSegmentationFilter<TInputImage, TOutputImage>::GenerateData()
{
  if (m_LabelImage == nullptr) // Exit if SetLabelImage has not been called
  {
    std::cout << "Label image has not been set" << std::endl;
    return;
  }
  if (OutputImageType::ImageDimension != 2 && OutputImageType::ImageDimension != 3)
  {
    std::cout << "Exit segmentation. Image dimension must be 2 or 3 but is " << OutputImageType::ImageDimension << std::endl;
    return;
  }

  // Get regions to iterate for original and label images
  typename OutputImageType::RegionType regionLabel = m_LabelImage->GetLargestPossibleRegion();

  // Set label image iterator to get the bounding box size and the quantity of marked nodes
  typedef itk::ImageRegionIterator<OutputImageType> IteratorLabelType;
  IteratorLabelType itLabel(m_LabelImage, regionLabel);

  // Crop image to bounds containin labels
  typename OutputImageType::RegionType regionLabelCrop;
  typename InputImageType::RegionType regionCrop;

  for (int i = 0; i != OutputImageType::ImageDimension; ++i)
  {
    regionLabelCrop.SetSize(i, regionLabel.GetIndex()[i]);
    regionLabelCrop.SetIndex(i, regionLabel.GetSize()[i]);
  }

  int markedLength = 0;

  itLabel.GoToBegin();
  while (!itLabel.IsAtEnd())
  {
    // Get image boundaries to crop image
    if (itLabel.Get() != 0)
    {
      ++markedLength;
      // Get bounding region
      for (int i = 0; i != OutputImageType::ImageDimension; ++i)
      {
        if (itLabel.GetIndex()[i] < regionLabelCrop.GetIndex()[i])
        {
          regionCrop.SetIndex(i, itLabel.GetIndex()[i]);
          regionLabelCrop.SetIndex(i, itLabel.GetIndex()[i]);
        }
        else if (itLabel.GetIndex()[i] > regionLabelCrop.GetSize()[i])
        {
          regionCrop.SetSize(i, itLabel.GetIndex()[i]);
          regionLabelCrop.SetSize(i, itLabel.GetIndex()[i]);
        }
      }
    }
    ++itLabel;
  }

  int totalNodes = 1;
  for (int i = 0; i != OutputImageType::ImageDimension; ++i)
  {
    regionCrop.SetIndex(i, regionCrop.GetIndex()[i]);
    regionLabelCrop.SetIndex(i, regionLabelCrop.GetIndex()[i]);
    regionCrop.SetSize(i, regionCrop.GetSize()[i] - regionCrop.GetIndex()[i] + 1);
    regionLabelCrop.SetSize(i, regionLabelCrop.GetSize()[i] - regionLabelCrop.GetIndex()[i] + 1);

    totalNodes *= regionCrop.GetSize()[i];
  }

  /////////////////////// Build Graph /////////////////////////////
  /*
    // Create graph. Each node will correspond to a pixel connected to its neighbors by an edge.
    // Neighbors are those nodes which distance is 1 ( d = sqrt((x-xi)² + (y-yi)² + (z-zi)²) = 1 ), 
    // where x, y, and z are the indices of a pixel and xi, yi, and zi are the indeces of a neighboring pixel
    // Iterate through all pixels of input and label images
    */

  int unmarkedLength = totalNodes - markedLength; /* Quantity of unmarked nodes */

  // Define vectors to store graph data
  std::vector<float> *nodes = new std::vector<float>(totalNodes);  /* Pixel intensity for all nodes/pixels */
  std::vector<float> *labels = new std::vector<float>(totalNodes); /* Label of each node/pixel */
  /*  For node 'i'. If 'i' is a marked node, previousFound->at(i) is how many unmarked nodes there are before node 'i'.
        If 'i' is an unmarked node, previousFound->at(i) is how many marked nodes there are before node 'i'.
        This values are needed to build ordered Lu and BT matrices.
    */
  std::vector<int> *previousFound = new std::vector<int>(totalNodes);
  std::vector<int> *unmarked = new std::vector<int>(unmarkedLength); /* Indices of unmarked nodes ordered */
  std::vector<int> *marked = new std::vector<int>(markedLength);     /* Store labels of marked nodes */
  std::vector<int> *markedIdx = new std::vector<int>(markedLength);  /* Indices of marked nodes ordered */
  std::vector<int> *nameLabels = new std::vector<int>();             /* Values of the different labels of the prior */

  // Set bounding box image iterators
  typedef itk::ImageRegionConstIterator<InputImageType> ConstIteratorImageType;
  ConstIteratorImageType itImageCrop(this->GetInput(), regionCrop);
  IteratorLabelType itLabelCrop(m_LabelImage, regionLabelCrop);

  int foundMarked = 0;
  int foundUnmarked = 0;
  int NodeIdx = 0;

  itImageCrop.GoToBegin();
  itLabelCrop.GoToBegin();
  while (!itImageCrop.IsAtEnd())
  {
    // Store intensity in a std::vector
    nodes->at(NodeIdx) = itImageCrop.Get();
    // Store labels in a std::vector. Each index of 'nodes' and 'labels' correspond to the same pixel
    labels->at(NodeIdx) = itLabelCrop.Get();

    if (itLabelCrop.Get() == 0)
    {
      // Get the index of each node that is unmarked
      unmarked->at(foundUnmarked) = NodeIdx;
      // Store how many marked points have been found before this unmarked node
      previousFound->at(NodeIdx) = foundMarked;
      ++foundUnmarked;
    }
    else
    {
      // Set to label value if label != 0
      marked->at(foundMarked) = itLabelCrop.Get();
      // Get the index of each node that is marked
      markedIdx->at(foundMarked) = NodeIdx;
      // Store how many unmarked points have been found before this marked node
      previousFound->at(NodeIdx) = foundUnmarked;
      ++foundMarked;
      bool found;

      // Store the different labels in a std::vector
      found = std::find(nameLabels->begin(), nameLabels->end(), itLabelCrop.Get()) != nameLabels->end();
      if (!found)
      {
        nameLabels->push_back(itLabelCrop.Get());
      }
    }
    ++itImageCrop;
    ++itLabelCrop;
    ++NodeIdx;
  }

  // Sort labels
  std::sort(nameLabels->begin(), nameLabels->end());

  int totalLabels = nameLabels->size();
  if (m_SolveForAllLabels)
  {
    totalLabels += 1;
  }
  // Linear system: Lu * X = -BT * M
  // Convert marked (M) into a Eigen::Sparse matrix markedRHS. Needed for the computation of -BT * M (Eigen::SparseMatrix * Eigen::SparseMatrix)
  Eigen::SparseMatrix<float, Eigen::ColMajor> *markedRHS = new Eigen::SparseMatrix<float, Eigen::ColMajor>(markedLength, totalLabels - 1);
  for (int i = 0; i != totalLabels - 1; ++i)
  {
    int pos = 0;
    for (auto itMarked = marked->begin(); itMarked != marked->end(); ++itMarked)
    {
      if (*itMarked == nameLabels->at(i))
        markedRHS->insert(pos, i) = 1;
      ++pos;
    }
  }

  marked->clear();
  delete marked;

  /////////////////////// Build Laplacian matrix /////////////////////////////
  // Normalize intensity gradient over image spacing
  std::vector<float> spacing;
  typename InputImageType::SpacingType space = this->GetInput()->GetSpacing();
  if (OutputImageType::ImageDimension == 2)
    spacing = {space[1], space[0], space[0], space[1]};
  else if (OutputImageType::ImageDimension == 3)
    spacing = {space[2], space[1], space[0], space[0], space[1], space[2]};

  std::vector<int> neighbors;
  int x = regionCrop.GetSize()[0];
  int y = regionCrop.GetSize()[1];
  //  Right hand of the equation: -BT * M
  Eigen::SparseMatrix<float, Eigen::ColMajor> *BTxM = new Eigen::SparseMatrix<float, Eigen::ColMajor>(markedLength, totalLabels - 1);

  // Build BT
  // Compare whether NumCols < NumRows for less iterations during BT building
  if (markedIdx->size() < unmarked->size())
  {
    Eigen::SparseMatrix<float, Eigen::ColMajor> *BT = new Eigen::SparseMatrix<float, Eigen::ColMajor>(unmarkedLength, markedLength);
    BT->reserve(Eigen::VectorXi::Constant(markedLength, 6));

    int node;
    float valNode, valNeighbor, w;

    // Iterate through marked nodes to build BT. Rows correspond to unmarked nodes, columns to marked nodes
    for (auto itMarked = markedIdx->begin(); itMarked != markedIdx->end(); ++itMarked)
    {
      valNode = nodes->at(*itMarked); // Intensity of node
      // Obtain neighbors indexes. right and left, top and bottom, front and back.
      node = *itMarked;
      if (OutputImageType::ImageDimension == 2)
        neighbors = {node - x, node - 1, node + 1, node + x};
      else if (OutputImageType::ImageDimension == 3)
        neighbors = {node - x * y, node - x, node - 1, node + 1, node + x, node + x * y};
      for (int i = 0; i != neighbors.size(); ++i)
      {
        // Make sure all the neighbors computed fall within the bounding box dimension
        if (neighbors.at(i) >= 0 && neighbors.at(i) < totalNodes && labels->at(neighbors.at(i)) == 0)
        {
          valNeighbor = nodes->at(neighbors.at(i));                                    // Intensity of neighbor pixel
          w = (exp(-m_Beta * pow((valNode - valNeighbor) / spacing.at(i), 2)) + 1e-6); // Intensity gradient following a Gaussian function
          //  Columns of BT correspond to marked nodes, rows to unmarked
          BT->insert(neighbors.at(i) - previousFound->at(neighbors.at(i)), node - previousFound->at(node)) = -w;
        }
      }
    }
    markedIdx->clear();
    //  Right hand of the equation: -BT * M
    *BTxM = -*BT * *markedRHS;
    BT->resize(0, 0);
    BT->data().squeeze();
    markedRHS->resize(0, 0);
    markedRHS->data().squeeze();
    delete markedIdx, markedRHS, BT;
  }
  else
  {
    markedIdx->clear();
    delete markedIdx;

    Eigen::SparseMatrix<float, Eigen::RowMajor> *BT = new Eigen::SparseMatrix<float, Eigen::RowMajor>(unmarkedLength, markedLength);
    BT->reserve(Eigen::VectorXi::Constant(markedLength, 6));

    int node;
    float valNode, valNeighbor, w;

    // Iterate through unmarked nodes to build BT. Rows correspond to unmarked nodes, columns to marked nodes
    for (auto itUnmarked = unmarked->begin(); itUnmarked != unmarked->end(); ++itUnmarked)
    {
      valNode = nodes->at(*itUnmarked); // Intensity of node
      // Obtain neighbors indexes. right and left, top and bottom, front and back.
      node = *itUnmarked;
      if (OutputImageType::ImageDimension == 2)
        neighbors = {node - x, node - 1, node + 1, node + x};
      else if (OutputImageType::ImageDimension == 3)
        neighbors = {node - x * y, node - x, node - 1, node + 1, node + x, node + x * y};
      for (int i = 0; i != neighbors.size(); ++i)
      {
        // Make sure all the neighbors computed fall within the bounding box dimension
        if (neighbors.at(i) >= 0 && neighbors.at(i) < totalNodes && labels->at(neighbors.at(i)) != 0)
        {
          valNeighbor = nodes->at(neighbors.at(i));                                    // Intensity of neighbor pixel
          w = (exp(-m_Beta * pow((valNode - valNeighbor) / spacing.at(i), 2)) + 1e-6); // Intensity gradient following a Gaussian function
          //  Columns of BT correspond to marked nodes, rows to unmarked
          BT->insert(node - previousFound->at(node), neighbors.at(i) - previousFound->at(neighbors.at(i))) = -w;
        }
      }
    }
    //  Right hand of the equation: -BT * M
    *BTxM = -*BT * *markedRHS;
    BT->resize(0, 0);
    BT->data().squeeze();
    markedRHS->resize(0, 0);
    markedRHS->data().squeeze();
    delete markedRHS, BT;
  }

  int nnz = 0;

  // Build Lu. LHS of the equation
  Eigen::SparseMatrix<float, Eigen::RowMajor> *Lu = new Eigen::SparseMatrix<float, Eigen::RowMajor>(unmarkedLength, unmarkedLength);
  Lu->reserve(Eigen::VectorXi::Constant(unmarkedLength, 7));

  // Iterate through unmarked nodes to build Lu. Rows correspond to unmarked nodes, columns to unmarked nodes
  for (auto itUnmarked = unmarked->begin(); itUnmarked != unmarked->end(); ++itUnmarked)
  {
    int node;
    float valNode, valNeighbor, w, degree;

    valNode = nodes->at(*itUnmarked); // Intensity of node
    // Obtain neighbors indexes. right and left, top and bottom, front and back.
    node = *itUnmarked;
    if (OutputImageType::ImageDimension == 2)
      neighbors = {node - x, node - 1, node + 1, node + x};
    else if (OutputImageType::ImageDimension == 3)
      neighbors = {node - x * y, node - x, node - 1, node + 1, node + x, node + x * y};
    degree = 0.0;
    for (int i = 0; i != neighbors.size(); ++i)
    {
      // Make sure all the neighbors computed fall within the bounding box dimension
      if (neighbors.at(i) >= 0 && neighbors.at(i) < totalNodes)
      {
        valNeighbor = nodes->at(neighbors.at(i));                                    // Intensity of neighbor pixel
        w = (exp(-m_Beta * pow((valNode - valNeighbor) / spacing.at(i), 2)) + 1e-6); // Intensity gradient following a Gaussian function
        degree += w;                                                                 // Sum of the weights
        //  Columns of Lu correspond to unmarked nodes
        if (labels->at(neighbors.at(i)) == 0) // If neighbor is an unmarked node, build Lu
        {
          Lu->insert(node - previousFound->at(node), neighbors.at(i) - previousFound->at(neighbors.at(i))) = -w;
          ++nnz;
        }
      }
    }
    // Add node degree to diagonal of Lu
    Lu->insert(node - previousFound->at(node), node - previousFound->at(node)) = degree;
    ++nnz;
  }
  Lu->makeCompressed();

  Eigen::SparseMatrix<float, Eigen::RowMajor> *Lu_PreconditionerGpu = new Eigen::SparseMatrix<float, Eigen::RowMajor>(unmarkedLength, unmarkedLength);
  Lu_PreconditionerGpu->reserve(Eigen::VectorXi::Constant(unmarkedLength, 1));
  for (int i = 0; i != unmarkedLength; ++i)
    Lu_PreconditionerGpu->insert(i, i) = 1 / Lu->coeffRef(i, i);
  Lu_PreconditionerGpu->makeCompressed();

  nodes->clear();
  unmarked->clear();
  previousFound->clear();
  neighbors.clear();
  delete nodes, unmarked, previousFound;

  // /////////////////////// Solve linear system /////////////////////////////
  // Convert BTxM into a float array for bicgstab cuda solver
  float *bhost_all_labels = new float[unmarkedLength * (totalLabels - 1)];

  for (int i = 0; i != unmarkedLength * (totalLabels - 1); ++i)
    bhost_all_labels[i] = 0;

  for (int k = 0; k < BTxM->outerSize(); ++k)
  {
    for (Eigen::SparseMatrix<float, Eigen::ColMajor>::InnerIterator itMat(*BTxM, k); itMat; ++itMat)
    {
      bhost_all_labels[itMat.row() + k * unmarkedLength] = itMat.value();
    }
  }

  // Get pointers for CSR matrices Lu and Lu_PreconditionerGpu
  cooValAhost = Lu->valuePtr();
  csrColPtrAhost = Lu->outerIndexPtr();
  cooRowPtrAhost = Lu->innerIndexPtr();

  cooValAhostM = Lu_PreconditionerGpu->valuePtr();
  csrColPtrAhostM = Lu_PreconditionerGpu->outerIndexPtr();
  cooRowPtrAhostM = Lu_PreconditionerGpu->innerIndexPtr();

  int bicgstab_exit_status, cuda_mem_cpy_exit_status, cuda_mem_free_exit_status;

  // Result vector
  probabilities = new float[unmarkedLength * (totalLabels - 1)];
  probabilitiesInner = new float[unmarkedLength];

  // Allocate GPU memory
  cuda_mem_cpy_exit_status = this->AllocGPUMemory(unmarkedLength, nnz);
  if (cuda_mem_cpy_exit_status != 0)
  {
    std::cout << "GPU memory allocation failed with error " << cuda_mem_cpy_exit_status << std::endl;
    return;
  }

  // Solve S-1 linear systems
  for (int linearSystem = 0; linearSystem != totalLabels - 1; ++linearSystem)
  {
    // Get pinter for each linear system RHS
    bhost = &bhost_all_labels[linearSystem * unmarkedLength];
    // Call CUDA BiCGStab solver
    bicgstab_exit_status = this->BiCGStab(unmarkedLength, nnz);
    if (bicgstab_exit_status != 0)
    {
      std::cout << "BiCStab failed with error" << bicgstab_exit_status << std::endl;
      return;
    }

    for (int i = 0; i != unmarkedLength; ++i)
      probabilities[i + linearSystem * unmarkedLength] = probabilitiesInner[i];
  }

  // Free GPU memeory when all linear systems have been solved
  cuda_mem_free_exit_status = this->FreeGPUMemory();
  if (cuda_mem_free_exit_status != 0)
  {
    std::cout << "GPU memory free failed with error" << cuda_mem_free_exit_status << std::endl;
  }
  Lu->resize(0, 0);
  Lu->data().squeeze();
  Lu_PreconditionerGpu->resize(0, 0);
  Lu_PreconditionerGpu->data().squeeze();
  BTxM->resize(0, 0);
  BTxM->data().squeeze();
  delete Lu, Lu_PreconditionerGpu, BTxM;
  delete[] probabilitiesInner, bhost, bhost_all_labels, cooRowPtrAhost, csrColPtrAhost, cooValAhost, cooRowPtrAhostM, csrColPtrAhostM, cooValAhostM;

  std::vector<int> *RWLabels = new std::vector<int>(unmarkedLength);

  /*  Assign a label to each unmarked node according to the result of the solver. 
      The label that is assigned is that one corresponding to the highest probability.
      Since we solver for S-1 systems, last label probability is computed by subtraction */
  for (int i = 0; i != unmarkedLength; ++i)
  {
    float maxProbability = probabilities[i];
    int maxLabelPos = 0;
    float accumulatedProbability = maxProbability;
    for (int j = 1; j != totalLabels - 1; ++j)
    {
      accumulatedProbability += probabilities[i + j * unmarkedLength];
      if (probabilities[i + j * unmarkedLength] > maxProbability)
      {
        maxProbability = probabilities[i + j * unmarkedLength];
        maxLabelPos = j;
      }
    }
    RWLabels->at(i) = 0.95 - accumulatedProbability > maxProbability ? nameLabels->back() : nameLabels->at(maxLabelPos);
  }

  delete[] probabilities;

  int valBackground;
  if (!m_WriteBackground)
    valBackground = 0;
  else
    valBackground = nameLabels->back();

  typename OutputImageType::Pointer outputLabels = this->GetOutput();

  outputLabels->Graft(m_LabelImage);
  outputLabels->FillBuffer(valBackground);

  // Iterate through original label image to create segmentation image
  // Assign labels known from label image to their original value. Assign unmarked labels
  // according to the result from the solver

  IteratorLabelType itOut1(outputLabels, regionLabelCrop);

  // Set computed labels to segmentation image
  int unmarkedIdx = 0;
  int idxOutput = 0;

  itOut1.GoToBegin();
  while (!itOut1.IsAtEnd())
  {
    if (labels->at(idxOutput) == 0)
    {
      if (RWLabels->at(unmarkedIdx) != nameLabels->back())
        itOut1.Set(RWLabels->at(unmarkedIdx));
      else
        itOut1.Set(valBackground);
      ++unmarkedIdx;
    }
    else if (labels->at(idxOutput) != nameLabels->back())
      itOut1.Set(labels->at(idxOutput));

    ++idxOutput;
    ++itOut1;
  }

  labels->clear();
  RWLabels->clear();
  nameLabels->clear();
  delete labels, RWLabels, nameLabels;

  return;
}

template <typename TInputImage, typename TOutputImage>
int CudaRWSegmentationFilter<TInputImage, TOutputImage>::BiCGStab(int M, int nnz)
{

  int iter, flag;

  cusparseStatus_t status;
  cusparseHandle_t handle = 0;
  cusparseMatDescr_t descra = 0;

  int nnzM = M;
  int N = M;
  int N2 = N;

  float timesOne[1] = {1.0};
  float timesZero[1] = {0.0};
  float timesMinusOne[1] = {-1.0};

  cudaStat4 = cudaMemcpy(r, bhost,
                         (size_t)(N * sizeof(r[0])),
                         cudaMemcpyHostToDevice);

  cudaStat5 = cudaMemcpy(xdev, xhost,
                         (size_t)(N * sizeof(xdev[0])),
                         cudaMemcpyHostToDevice);

  // Copy right hand side to GPU memory
  if ((cudaStat4 != cudaSuccess) ||
      (cudaStat5 != cudaSuccess))
  {
    CLEANUP("Memcpy from Host to Device failed");
    return EXIT_FAILURE;
  }

  /* initialize cusparse library */
  status = cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    CLEANUP("CUSPARSE Library initialization failed");
    return EXIT_FAILURE;
  }
  /* create and setup matrix descriptor */
  status = cusparseCreateMatDescr(&descra);
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    CLEANUP("Matrix descriptor initialization failed");
    return EXIT_FAILURE;
  }
  cusparseSetMatType(descra, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ZERO);

  bnrm2 = cublasSnrm2(N, r, 1);

  float tol = m_Tolerance * bnrm2;

  status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnz, timesMinusOne, descra, cooValAdev, csrColPtrAdev, cooRowPtrAdev, xdev, timesOne, r); /* r = r - A*x; r is now residual */

  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    CLEANUP("Matrix‐vector multiplication failed");
    return EXIT_FAILURE;
  }

  error = cublasSnrm2(N, r, 1) / bnrm2; /* norm_r = norm(b) */
  if (error < tol)
  {
    CLEANUP("Error smaller than tolerance failed");
    return EXIT_FAILURE; /* x is close enough already */
  }

  omega = 1.0;

  cudaStat1 = cudaMemcpy(r_tld, r, (size_t)(N * sizeof(r[0])),
                         cudaMemcpyDeviceToDevice); /* r_tld = r */

  if ((cudaStat1 != cudaSuccess))
  {
    CLEANUP("Memcpy from r to r_tld failed");
    return EXIT_FAILURE;
  }

  // Loop of the BiCGStab solver
  for (iter = 0; iter < m_MaximumNumberOfIterations; ++iter)
  {
    rho = cublasSdot(N, r_tld, 1, r, 1); /* rho = r_tld'*r */

    if (rho == 0.0)
      break;

    if (iter > 0)
    {
      beta = (rho / rho_1) * (alpha / omega);
      cublasSaxpy(N, -omega, v, 1, p, 1);
      cublasSaxpy(N, 1.0 / beta, r, 1, p, 1);
      cublasSscal(N, beta, p, 1); /* p = r + beta*( p - omega*v ) */
    }
    else
    {
      cudaStat1 = cudaMemcpy(p, r,
                             (size_t)(N * sizeof(r[0])),
                             cudaMemcpyDeviceToDevice); /* p = r */
      if ((cudaStat1 != cudaSuccess))
      {
        CLEANUP("Memcpy from r to p failed");
        return EXIT_FAILURE;
      }
    }

    status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N2, N2, nnzM, timesOne, descra, cooValAdevM, csrColPtrAdevM, cooRowPtrAdevM, p, timesZero, p_hat);

    status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnz, timesOne, descra, cooValAdev, csrColPtrAdev, cooRowPtrAdev, p_hat, timesZero, v); /* v = A*p_hat */

    alpha = rho / cublasSdot(N, r_tld, 1, v, 1); /* alph = rho / ( r_tld'*v ) */

    cudaStat1 = cudaMemcpy(s, r, (size_t)(N * sizeof(r[0])),
                           cudaMemcpyDeviceToDevice); /* s = r */
    if ((cudaStat1 != cudaSuccess))
    {
      CLEANUP("Memcpy from r to s failed");
      return EXIT_FAILURE;
    }
    cublasSaxpy(N, -alpha, v, 1, s, 1);
    snrm2 = cublasSnrm2(N, s, 1);

    cublasSaxpy(N, alpha, p_hat, 1, xdev, 1); /*  h = x + alph*p_hat */

    if (snrm2 < tol)
    {
      // cublasSaxpy(N, alpha, p_hat, 1, s, 1);
      resid = snrm2 / bnrm2;

      cudaStat5 = cudaMemcpy(probabilitiesInner, xdev,
                             (size_t)(N * sizeof(probabilitiesInner[0])),
                             cudaMemcpyDeviceToHost); /* x = h */
      if ((cudaStat5 != cudaSuccess))
      {
        CLEANUP("Memcpy from x to xhost failed");
        return EXIT_FAILURE;
      }

      break;
    }

    status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N2, N2, nnzM, timesOne, descra, cooValAdevM, csrColPtrAdevM, cooRowPtrAdevM, s, timesZero, s_hat);

    status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nnz, timesOne, descra, cooValAdev, csrColPtrAdev, cooRowPtrAdev, s_hat, timesZero, t); /* t = A*s_hat */

    omega = cublasSdot(N, t, 1, s, 1) / cublasSdot(N, t, 1, t, 1); /* omega = ( t'*s) / ( t'*t ) */

    cublasSaxpy(N, omega, s_hat, 1, xdev, 1); /*  x = h + omega*s_hat */

    cudaStat1 = cudaMemcpy(r, s, (size_t)(N * sizeof(r[0])),
                           cudaMemcpyDeviceToDevice); /* r = s */
    if ((cudaStat1 != cudaSuccess))
    {
      CLEANUP("Memcpy from s to r failed");
      return EXIT_FAILURE;
    }

    cublasSaxpy(N, -omega, t, 1, r, 1); /* r = s - omega*t */
    snrm2 = cublasSnrm2(N, r, 1);

    if (snrm2 <= tol)
    {
      resid = snrm2 / bnrm2;

      cudaStat5 = cudaMemcpy(probabilitiesInner, xdev,
                             (size_t)(N * sizeof(probabilitiesInner[0])),
                             cudaMemcpyDeviceToHost); /* x = x */
      if ((cudaStat5 != cudaSuccess))
      {
        CLEANUP("Memcpy from x to xhost failed");
        return EXIT_FAILURE;
      }
      break;
    }
    if (omega == 0.0)
      break;
    rho_1 = rho;
  }
  ++iter;

  error = snrm2 / bnrm2;

  if (error <= tol)
  {
    flag = 0;
  }
  else if (omega == 0.0)
  {
    flag = -2;
  }
  else if (rho == 0.0)
  {
    flag = -1;
  }
  else
    flag = 1;

  m_SolverIterations += iter;
  m_SolverError = std::min(m_SolverError, error);

  return flag;
}

// Allocate GPU memory and copy data to device
template <typename TInputImage, typename TOutputImage>
int CudaRWSegmentationFilter<TInputImage, TOutputImage>::AllocGPUMemory(int M, int nnz)
{

  int nnzM = M; // Lu has size MxM, Lu_PreconditionerGpu has size MxM
  int N = M;

  // Define initial guess
  xhost = new float[N];
  for (int i = 0; i < N; i++)
  {
    xhost[i] = 0;
  }

  cudaStat1 = cudaMalloc((void **)&cooRowPtrAdev, nnz * sizeof(cooRowPtrAdev[0]));
  cudaStat3 = cudaMalloc((void **)&cooValAdev, nnz * sizeof(cooValAdev[0]));
  cudaStat4 = cudaMalloc((void **)&r, N * sizeof(r[0]));
  cudaStat5 = cudaMalloc((void **)&s, N * sizeof(s[0]));
  cudaStat6 = cudaMalloc((void **)&t, N * sizeof(t[0]));
  cudaStat7 = cudaMalloc((void **)&r_tld, N * sizeof(r_tld[0]));
  cudaStat8 = cudaMalloc((void **)&p, N * sizeof(p[0]));
  cudaStat9 = cudaMalloc((void **)&p_hat, N * sizeof(p_hat[0]));
  cudaStat10 = cudaMalloc((void **)&s_hat, N * sizeof(s_hat[0]));
  cudaStat11 = cudaMalloc((void **)&v, N * sizeof(v[0]));
  cudaStat12 = cudaMalloc((void **)&xdev, N * sizeof(xdev[0]));

  cudaStat13 = cudaMalloc((void **)&csrColPtrAdev, (N + 1) * sizeof(csrColPtrAdev[0]));

  cudaStat14 = cudaMalloc((void **)&cooRowPtrAdevM, nnzM * sizeof(cooRowPtrAdevM[0]));
  cudaStat16 = cudaMalloc((void **)&cooValAdevM, nnzM * sizeof(cooValAdevM[0]));
  cudaStat17 = cudaMalloc((void **)&csrColPtrAdevM, (N + 1) * sizeof(csrColPtrAdevM[0]));

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ||
      (cudaStat5 != cudaSuccess) ||
      (cudaStat6 != cudaSuccess) ||
      (cudaStat7 != cudaSuccess) ||
      (cudaStat8 != cudaSuccess) ||
      (cudaStat9 != cudaSuccess) ||
      (cudaStat10 != cudaSuccess) ||
      (cudaStat11 != cudaSuccess) ||
      (cudaStat12 != cudaSuccess) ||

      (cudaStat14 != cudaSuccess) ||
      (cudaStat16 != cudaSuccess) ||
      (cudaStat17 != cudaSuccess))
  {
    CLEANUP("Device malloc failed");
    return EXIT_FAILURE;
  }

  cudaStat1 = cudaMemcpy(cooRowPtrAdev, cooRowPtrAhost,
                         (size_t)(nnz * sizeof(cooRowPtrAdev[0])),
                         cudaMemcpyHostToDevice);
  cudaStat13 = cudaMemcpy(csrColPtrAdev, csrColPtrAhost,
                          (size_t)((N + 1) * sizeof(csrColPtrAdev[0])),
                          cudaMemcpyHostToDevice);
  cudaStat3 = cudaMemcpy(cooValAdev, cooValAhost,
                         (size_t)(nnz * sizeof(cooValAdev[0])),
                         cudaMemcpyHostToDevice);

  cudaStat14 = cudaMemcpy(cooRowPtrAdevM, cooRowPtrAhostM,
                          (size_t)(nnzM * sizeof(cooRowPtrAdevM[0])),
                          cudaMemcpyHostToDevice);
  cudaStat17 = cudaMemcpy(csrColPtrAdevM, csrColPtrAhostM,
                          (size_t)((N + 1) * sizeof(cooColPtrAdevM[0])),
                          cudaMemcpyHostToDevice);
  cudaStat16 = cudaMemcpy(cooValAdevM, cooValAhostM,
                          (size_t)(nnzM * sizeof(cooValAdevM[0])),
                          cudaMemcpyHostToDevice);

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat13 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat14 != cudaSuccess) ||
      (cudaStat17 != cudaSuccess) ||
      (cudaStat16 != cudaSuccess))
  {
    CLEANUP("Memcpy from Host to Device failed");
    return EXIT_FAILURE;
  }

  return 0;
}

// Free GPU memeory when all linear systems have been solved
template <typename TInputImage, typename TOutputImage>
int CudaRWSegmentationFilter<TInputImage, TOutputImage>::FreeGPUMemory()
{
  /* shutdown CUBLAS */
  cublas_status = cublasShutdown();
  if (cublas_status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }

  cudaStat1 = cudaFree(cooRowPtrAdev);
  cudaStat3 = cudaFree(cooValAdev);
  cudaStat4 = cudaFree(r);
  cudaStat5 = cudaFree(s);
  cudaStat6 = cudaFree(t);
  cudaStat7 = cudaFree(r_tld);
  cudaStat8 = cudaFree(p);
  cudaStat9 = cudaFree(p_hat);
  cudaStat10 = cudaFree(s_hat);
  cudaStat11 = cudaFree(v);
  cudaStat12 = cudaFree(xdev);

  cudaStat13 = cudaFree(csrColPtrAdev);

  cudaStat14 = cudaFree(cooRowPtrAdevM);
  cudaStat16 = cudaFree(cooValAdevM);
  cudaStat17 = cudaFree(csrColPtrAdevM);

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ||
      (cudaStat5 != cudaSuccess) ||
      (cudaStat6 != cudaSuccess) ||
      (cudaStat7 != cudaSuccess) ||
      (cudaStat8 != cudaSuccess) ||
      (cudaStat9 != cudaSuccess) ||
      (cudaStat10 != cudaSuccess) ||
      (cudaStat11 != cudaSuccess) ||
      (cudaStat12 != cudaSuccess) ||

      (cudaStat14 != cudaSuccess) ||
      (cudaStat16 != cudaSuccess) ||
      (cudaStat17 != cudaSuccess))
  {
    CLEANUP("Device memory free failed");
    return EXIT_FAILURE;
  }

  return 0;
}

template <typename TInputImage, typename TOutputImage>
void CudaRWSegmentationFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Beta: " << m_Beta << std::endl;
  os << indent << "Tolerance: " << m_Tolerance << std::endl;
  os << indent << "MaximumNumberOfIterations: " << m_MaximumNumberOfIterations << std::endl;
  os << indent << "WriteBackground: " << m_WriteBackground << std::endl;
}

} // namespace itk
#endif