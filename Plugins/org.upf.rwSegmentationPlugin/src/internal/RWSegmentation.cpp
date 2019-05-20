/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "RWSegmentation.h"

// Qt
#include <QMessageBox>

//mitk image
#include <mitkImage.h>
#include "mitkImageCast.h"
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateProperty.h>

// ITK
#include <itkImageRegionIterator.h>

// RW filters
#include "itkRWSegmentationFilter/itkRWSegmentationFilter.h"
#if CUDA_IS_FOUND
#include "itkCudaRWSegmentationFilter/itkCudaRWSegmentationFilter.h"
#endif

const std::string RWSegmentation::VIEW_ID = "org.mitk.views.rwsegmentation";

void RWSegmentation::SetFocus()
{
  m_Controls.DoSegmentation->setFocus();
}

void RWSegmentation::CreateQtPartControl(QWidget *parent)
{
  // create GUI widgets from the Qt Designer's .ui file
  m_Controls.setupUi(parent);

#if CUDA_IS_FOUND
  m_Controls.gpuRadioButton->setEnabled(true);
  MITK_INFO << "GPU enabled";
#else
  m_Controls.gpuRadioButton->setEnabled(false);
  MITK_INFO << "CUDA not found. GPU disabled.";
#endif

  m_Controls.imageComboBox->SetDataStorage(this->GetDataStorage());
  m_Controls.imageComboBox->SetPredicate(mitk::TNodePredicateDataType<mitk::Image>::New());
  m_Controls.foregroundComboBox->SetDataStorage(this->GetDataStorage());
  m_Controls.foregroundComboBox->SetPredicate(mitk::TNodePredicateDataType<mitk::Image>::New());
  m_Controls.backgroundComboBox->SetDataStorage(this->GetDataStorage());
  m_Controls.backgroundComboBox->SetPredicate(mitk::TNodePredicateDataType<mitk::Image>::New());

  connect(m_Controls.DoSegmentation, SIGNAL(clicked()), this, SLOT(DoRWSegmentation()));
  connect(m_Controls.gpuRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnGpuRadioButtonToggled(bool)));

  this->OnGpuRadioButtonToggled(false);
}

void RWSegmentation::OnGpuRadioButtonToggled(bool checked)
{
  if (checked)
  {
    m_Controls.numThreadsSpinBox->setEnabled(false);
    m_Controls.threadsLabel->setEnabled(false);
  }
  else
  {
    m_Controls.numThreadsSpinBox->setEnabled(true);
    m_Controls.threadsLabel->setEnabled(true);
  }
}

// Do RW segmentation
void RWSegmentation::DoRWSegmentation()
{
  if (m_Controls.imageComboBox->GetSelectedNode() == nullptr ||
      m_Controls.imageComboBox->GetSelectedNode() == m_Controls.foregroundComboBox->GetSelectedNode() ||
      m_Controls.imageComboBox->GetSelectedNode() == m_Controls.backgroundComboBox->GetSelectedNode() ||
      m_Controls.backgroundComboBox->GetSelectedNode() == m_Controls.foregroundComboBox->GetSelectedNode())
  {
    MITK_INFO << "Please select the image, the foreground segmentation and the background segmentation and ensure they are the correct ones.";
    return;
  }
  mitk::DataNode::Pointer imageNode = m_Controls.imageComboBox->GetSelectedNode();
  mitk::Image::Pointer inputMitkImage = dynamic_cast<mitk::Image *>(imageNode->GetData());
  mitk::Image::Pointer labelMitkForegroundImage = dynamic_cast<mitk::Image *>(m_Controls.foregroundComboBox->GetSelectedNode()->GetData());
  mitk::Image::Pointer labelMitkBackgroundImage = dynamic_cast<mitk::Image *>(m_Controls.backgroundComboBox->GetSelectedNode()->GetData());

  InputImageType::Pointer inputItkImage = InputImageType::New();
  CastToItkImage(inputMitkImage, inputItkImage); //OK, now you can use inputItkImage whereever you want

  // Get label image
  LabelImageType::Pointer labelImage = RWSegmentation::MergeLabels(labelMitkForegroundImage, labelMitkBackgroundImage);

  // Define output image
  mitk::Image::Pointer outputImage = mitk::Image::New();

  int iterations;
  double error;
  if (m_Controls.cpuRadioButton->isChecked())
  {
    MITK_INFO << "RW with CPU";
    typedef itk::RWSegmentationFilter<InputImageType, LabelImageType> RWFilterType;
    RWFilterType::Pointer RWFilter = RWFilterType::New();
    RWFilter->SetInput(inputItkImage);
    RWFilter->SetLabelImage(labelImage);
    RWFilter->SetBeta(m_Controls.beta->value());
    RWFilter->SetNumberOfThreads(m_Controls.numThreadsSpinBox->value());
    RWFilter->WriteBackgroundOff();
    RWFilter->Update();

    iterations = RWFilter->GetSolverIterations();
    error = RWFilter->GetSolverError();

    //Get output image and convert to MITK image for display
    mitk::CastToMitkImage(RWFilter->GetOutput(), outputImage);
  }
  else if (m_Controls.gpuRadioButton->isChecked())
  {
#if CUDA_IS_FOUND
    MITK_INFO << "RW with GPU";
    typedef itk::CudaRWSegmentationFilter<InputImageType, LabelImageType> RWFilterType;
    RWFilterType::Pointer RWFilter = RWFilterType::New();
    RWFilter->SetInput(inputItkImage);
    RWFilter->SetLabelImage(labelImage);
    RWFilter->SetBeta(m_Controls.beta->value());
    RWFilter->WriteBackgroundOff();
    RWFilter->Update();

    iterations = RWFilter->GetSolverIterations();
    error = RWFilter->GetSolverError();

    //Get output image and convert to MITK image for display
    mitk::CastToMitkImage(RWFilter->GetOutput(), outputImage);
#endif
  }

  MITK_INFO << "BiCGStab finished with error " << error << " after " << iterations << " iterations.";
  // Display image as a new node
  mitk::DataNode::Pointer outputNode = mitk::DataNode::New();
  outputNode->SetData(outputImage);
  outputNode->SetName(imageNode->GetName() + "_RW_Segmentation");
  this->GetDataStorage()->Add(outputNode, imageNode);

  return;
}

LabelImageType::Pointer RWSegmentation::MergeLabels(mitk::Image::Pointer labelMitkForegroundImage, mitk::Image::Pointer labelMitkBackgroundImage)
{
  LabelImageType::Pointer labelForegroundImage = LabelImageType::New();
  LabelImageType::Pointer labelBackgroundImage = LabelImageType::New();

  CastToItkImage(labelMitkForegroundImage, labelForegroundImage); //OK, now you can use inputItkImage whereever you want
  CastToItkImage(labelMitkBackgroundImage, labelBackgroundImage); //OK, now you can use inputItkImage whereever you want

  typedef itk::ImageRegionConstIterator<LabelImageType> ConstIteratorImageType;
  typedef itk::ImageRegionIterator<LabelImageType> IteratorImageType;

  LabelImageType::RegionType region = labelForegroundImage->GetLargestPossibleRegion();
  LabelImageType::SpacingType spacing = labelForegroundImage->GetSpacing();
  LabelImageType::PointType origin = labelForegroundImage->GetOrigin();

  LabelImageType::RegionType region2 = labelBackgroundImage->GetLargestPossibleRegion();

  LabelImageType::Pointer outputLabels = LabelImageType::New();

  // Define output image (segmentation image) properties
  LabelImageType::RegionType outputRegion;
  LabelImageType::RegionType::IndexType outputStart;
  outputStart[0] = 0;
  outputStart[1] = 0;
  outputStart[2] = 0;

  LabelImageType::RegionType::SizeType size;
  size[0] = region.GetSize()[0];
  size[1] = region.GetSize()[1];
  size[2] = region.GetSize()[2];

  outputRegion.SetSize(size);
  outputRegion.SetIndex(outputStart);

  outputLabels->SetRegions(outputRegion);
  outputLabels->SetSpacing(spacing);
  outputLabels->SetOrigin(origin);
  outputLabels->Allocate();

  ConstIteratorImageType it1(labelForegroundImage, region);
  ConstIteratorImageType it2(labelBackgroundImage, region2);
  IteratorImageType it3(outputLabels, outputRegion);

  it1.GoToBegin();
  it2.GoToBegin();
  it3.GoToBegin();

  while (!it1.IsAtEnd())
  {
    if (it1.Get() == 1 && it2.Get() == 0)
      it3.Set(1);
    else if (it1.Get() == 0 && it2.Get() == 1)
      it3.Set(2);
    else if (it1.Get() != 0 && it2.Get() != 0)
      it3.Set(0);
    else
      it3.Set(0);

    ++it1;
    ++it2;
    ++it3;
  }

  return outputLabels;
}
