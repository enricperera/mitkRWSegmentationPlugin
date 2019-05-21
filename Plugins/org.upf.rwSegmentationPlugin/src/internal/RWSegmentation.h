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

#ifndef RWSegmentation_h
#define RWSegmentation_h

#include <berryISelectionListener.h>

#include <QmitkAbstractView.h>

#include "ui_RWSegmentationControls.h"

//mitk image
#include <mitkImage.h>
#include "itkImage.h"

/**
  \brief RWSegmentation
  Plugin for RW segmentation, which can be executed on CPU and on GPU (latter only if CUDA is installed)
*/

typedef float InputPixelType;
const int Dimension = 3;
typedef itk::Image<InputPixelType, Dimension> InputImageType;

typedef unsigned char LabelPixelType;
typedef itk::Image<LabelPixelType, Dimension> LabelImageType;

class RWSegmentation : public QmitkAbstractView
{
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:
  static const std::string VIEW_ID;

protected slots:

  /// \brief Called when the user clicks the GUI button
  void DoRWSegmentation();
  void OnGpuRadioButtonToggled(bool checked);
  void OnAddPushButton();
  void OnRemovePushButton();
  void OnRemoveAllPushButton();

protected:
  virtual void CreateQtPartControl(QWidget *parent) override;

  virtual void SetFocus() override;

  LabelImageType::Pointer ComputeLabelImage();

  Ui::RWSegmentationControls m_Controls;
};

#endif // RWSegmentation_h
