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

#ifndef itkRWSegmentationFilter_hxx
#define itkRWSegmentationFilter_hxx

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace itk
{
template <typename TInputImage, typename TOutputImage>
void RWSegmentationFilter<TInputImage, TOutputImage>::GenerateData()
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
        }
      }
    }
    // Add node degree to diagonal of Lu
    Lu->insert(node - previousFound->at(node), node - previousFound->at(node)) = degree;
  }
  Lu->makeCompressed();

  nodes->clear();
  unmarked->clear();
  previousFound->clear();
  neighbors.clear();
  delete nodes, unmarked, previousFound;

  /////////////////////// Solve linear system /////////////////////////////
  // Lu * X = -BT * M
  // Set vector to store the result of the solver
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> *probabilities = new Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(unmarkedLength, totalLabels - 1);

  // Allow multithreading. For the moment limited to 8 threads.
  if (m_NumberOfThreads > 0 && m_NumberOfThreads < 9)
  {
    omp_set_num_threads(m_NumberOfThreads);
    Eigen::setNbThreads(m_NumberOfThreads);
  }
  else
  {
    omp_set_num_threads(1);
    Eigen::setNbThreads(1);
  }

  // Set the solver for the problem with LHS Lu. BiCGStab.
  // Select either Eigen BiCGSTAB or ConjugateGradient
  Eigen::BiCGSTAB<Eigen::SparseMatrix<float, Eigen::RowMajor>, Eigen::DiagonalPreconditioner<float>> *solver = new Eigen::BiCGSTAB<Eigen::SparseMatrix<float, Eigen::RowMajor>, Eigen::DiagonalPreconditioner<float>>(*Lu); // Usually faster
  // Eigen::ConjugateGradient< Eigen::SparseMatrix< float ,Eigen::RowMajor> , Eigen::Lower|Eigen::Upper , Eigen::DiagonalPreconditioner<float> > solver(Lu);

  // Set solver parameters
  solver->setTolerance(m_Tolerance);
  solver->setMaxIterations(m_MaximumNumberOfIterations);
  // Compute probabilities with RHS BTxM
  *probabilities = solver->solve(*BTxM);

  m_SolverIterations = solver->iterations();
  m_SolverError = solver->error();

  Lu->resize(0, 0);
  Lu->data().squeeze();
  BTxM->resize(0, 0);
  BTxM->data().squeeze();
  delete Lu, BTxM;

  std::vector<int> *RWLabels = new std::vector<int>(unmarkedLength);

  /*  Assign a label to each unmarked node according to the result of the solver. 
        The label that is assigned is that one corresponding to the highest probability.
        Since we solver for S-1 systems, last label probability is computed by subtraction */
  for (int i = 0; i != unmarkedLength; ++i)
  {
    float maxProbability = probabilities->coeffRef(i, 0);
    int maxLabelPos = 0;
    float accumulatedProbability = maxProbability;
    for (int j = 1; j != totalLabels - 1; ++j)
    {
      accumulatedProbability += probabilities->coeffRef(i, j);
      if (probabilities->coeffRef(i, j) > maxProbability)
      {
        maxProbability = probabilities->coeffRef(i, j);
        maxLabelPos = j;
      }
    }
    RWLabels->at(i) = 0.95 - accumulatedProbability > maxProbability ? nameLabels->back() : nameLabels->at(maxLabelPos);
  }
  probabilities->resize(0, 0);
  delete probabilities;

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
void RWSegmentationFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Beta: " << m_Beta << std::endl;
  os << indent << "NumberOfThreads: " << m_NumberOfThreads << std::endl;
  os << indent << "Tolerance: " << m_Tolerance << std::endl;
  os << indent << "MaximumNumberOfIterations: " << m_MaximumNumberOfIterations << std::endl;
  os << indent << "WriteBackground: " << m_WriteBackground << std::endl;
}

} // namespace itk
#endif