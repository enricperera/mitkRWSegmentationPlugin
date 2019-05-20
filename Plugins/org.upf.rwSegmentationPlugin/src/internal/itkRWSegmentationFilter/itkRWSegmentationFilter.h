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

#ifndef itkRWSegmentationFilter_h
#define itkRWSegmentationFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"

namespace itk
{
template <typename TInputImage, typename TOutputImage>
class RWSegmentationFilter : public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RWSegmentationFilter Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RWSegmentationFilter, ImageToImageFilter);

  /** Image type information. */
  typedef TInputImage InputImageType;
  typedef TOutputImage OutputImageType;

  itkSetMacro(LabelImage, typename OutputImageType::Pointer);
  itkGetMacro(LabelImage, typename OutputImageType::Pointer);

  itkSetMacro(Beta, double);
  itkGetMacro(Beta, double);

  itkSetMacro(NumberOfThreads, int);
  itkGetMacro(NumberOfThreads, int);

  itkSetMacro(Tolerance, double);
  itkGetMacro(Tolerance, double);

  itkSetMacro(MaximumNumberOfIterations, int);
  itkGetMacro(MaximumNumberOfIterations, int);

  itkSetMacro(WriteBackground, bool);
  itkGetMacro(WriteBackground, bool);
  itkBooleanMacro(WriteBackground);

  itkSetMacro(SolveForAllLabels, bool);
  itkGetMacro(SolveForAllLabels, bool);
  itkBooleanMacro(SolveForAllLabels);

  itkGetMacro(SolverIterations, int);
  itkGetMacro(SolverError, float);

protected:
  RWSegmentationFilter() : m_LabelImage(nullptr), m_Beta(1), m_NumberOfThreads(1), m_Tolerance(1e-3),
                           m_MaximumNumberOfIterations(500), m_WriteBackground(true), m_SolveForAllLabels(false),
                           m_SolverIterations(0), m_SolverError(1) {}

  virtual ~RWSegmentationFilter() {}
  void PrintSelf(std::ostream &os, Indent indent) const ITK_OVERRIDE;

  /** Does the actual work */
  void GenerateData() ITK_OVERRIDE;

private:
  RWSegmentationFilter(const Self &) ITK_DELETE_FUNCTION;
  void operator=(const Self &) ITK_DELETE_FUNCTION;

  typename OutputImageType::Pointer m_LabelImage;
  double m_Beta;
  int m_NumberOfThreads;
  double m_Tolerance;
  int m_MaximumNumberOfIterations;
  bool m_WriteBackground;
  bool m_SolveForAllLabels;

  int m_SolverIterations;
  float m_SolverError;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRWSegmentationFilter.hxx"
#endif

#endif