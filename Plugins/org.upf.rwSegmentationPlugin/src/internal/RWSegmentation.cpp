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
#include "itkAddImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"

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

  connect(m_Controls.DoSegmentation, SIGNAL(clicked()), this, SLOT(DoRWSegmentation()));
  connect(m_Controls.addPushButton, SIGNAL(clicked()), this, SLOT(OnAddPushButton()));
  connect(m_Controls.removePushButton, SIGNAL(clicked()), this, SLOT(OnRemovePushButton()));
  connect(m_Controls.removeAllPushButton, SIGNAL(clicked()), this, SLOT(OnRemoveAllPushButton()));
  connect(m_Controls.gpuRadioButton, SIGNAL(toggled(bool)), this, SLOT(OnGpuRadioButtonToggled(bool)));

  this->OnGpuRadioButtonToggled(false);
}

void RWSegmentation::OnAddPushButton()
{
  QList<mitk::DataNode::Pointer> nodes = this->GetDataManagerSelection();
  if (nodes.empty())
    return;

  mitk::DataNode *node = nodes.front();
  if (!node)
    return;

  mitk::Image *image = dynamic_cast<mitk::Image *>(node->GetData());
  if (image)
  {
    m_Controls.labelImagesListWidget->addItem(QString::fromStdString(node->GetName()));
  }
}

void RWSegmentation::OnRemovePushButton()
{
  m_Controls.labelImagesListWidget->takeItem(m_Controls.labelImagesListWidget->currentRow());
}

void RWSegmentation::OnRemoveAllPushButton()
{
  m_Controls.labelImagesListWidget->clear();
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
  if (m_Controls.imageComboBox->GetSelectedNode() == nullptr)
  {
    MITK_INFO << "Please select an image.";
    return;
  }
  mitk::DataNode::Pointer imageNode = m_Controls.imageComboBox->GetSelectedNode();
  mitk::Image::Pointer inputMitkImage = dynamic_cast<mitk::Image *>(imageNode->GetData());

  InputImageType::Pointer inputItkImage = InputImageType::New();
  CastToItkImage(inputMitkImage, inputItkImage); //OK, now you can use inputItkImage whereever you want

  // Get label image
  LabelImageType::Pointer labelImage = this->ComputeLabelImage();
  if (labelImage == nullptr)
  {
    MITK_INFO << "Please select label images.";
    return;
  }

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

LabelImageType::Pointer RWSegmentation::ComputeLabelImage()
{
  int numberOfLabelImages = m_Controls.labelImagesListWidget->count();
  if (numberOfLabelImages < 1)
    return nullptr; // Return if no label images are selected
  else if (numberOfLabelImages == 1)
  {
    // Return the image itself, assuming it is an image that includes all the labels, with a different pixel value for each label
    QString nodeName = m_Controls.labelImagesListWidget->item(0)->text();
    mitk::DataNode::Pointer node = this->GetDataStorage()->GetNamedNode(nodeName.toStdString());
    mitk::Image::Pointer image1 = dynamic_cast<mitk::Image *>(node->GetData());

    LabelImageType::Pointer outputLabels = LabelImageType::New();
    CastToItkImage(image1, outputLabels);
    return outputLabels;
  }
  else
  {
    // Assume each image is a mask by itsel. Each image is binarized so all non-zero pixels are set to 0,1,...,N
    // The images are added into a single image, where all the labels will be included, each with a different pixel value.
    QString nodeName = m_Controls.labelImagesListWidget->item(0)->text();
    mitk::DataNode::Pointer node = this->GetDataStorage()->GetNamedNode(nodeName.toStdString());
    mitk::Image::Pointer image1 = dynamic_cast<mitk::Image *>(node->GetData());

    LabelImageType::Pointer outputLabels = LabelImageType::New();
    CastToItkImage(image1, outputLabels);

    // Binarize all images
    typedef itk::BinaryThresholdImageFilter<LabelImageType, LabelImageType> BinaryThresholdImageFilterType;
    BinaryThresholdImageFilterType::Pointer thresholdFilter1 = BinaryThresholdImageFilterType::New();
    thresholdFilter1->SetInput(outputLabels);
    thresholdFilter1->SetLowerThreshold(1);
    thresholdFilter1->SetUpperThreshold(255);
    thresholdFilter1->SetInsideValue(1);
    thresholdFilter1->SetOutsideValue(0);
    thresholdFilter1->Update();

    outputLabels = thresholdFilter1->GetOutput();

    for (int i = 1; i != numberOfLabelImages; ++i)
    {
      QString nodeName = m_Controls.labelImagesListWidget->item(i)->text();
      mitk::DataNode::Pointer node = this->GetDataStorage()->GetNamedNode(nodeName.toStdString());
      mitk::Image::Pointer image2 = dynamic_cast<mitk::Image *>(node->GetData());

      LabelImageType::Pointer itkLabelImage2 = LabelImageType::New();
      CastToItkImage(image2, itkLabelImage2); //OK, now you can use inputItkImage whereever you want

      BinaryThresholdImageFilterType::Pointer thresholdFilter2 = BinaryThresholdImageFilterType::New();
      thresholdFilter2->SetInput(itkLabelImage2);
      thresholdFilter2->SetLowerThreshold(1);
      thresholdFilter2->SetUpperThreshold(255);
      thresholdFilter2->SetInsideValue(i + 1);
      thresholdFilter2->SetOutsideValue(0);

      typedef itk::AddImageFilter<LabelImageType, LabelImageType> AddImageFilterType;
      AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
      addFilter->SetInput1(outputLabels);
      addFilter->SetInput2(thresholdFilter2->GetOutput());
      addFilter->Update();

      outputLabels = addFilter->GetOutput();
    }

    return outputLabels;
  }
}
