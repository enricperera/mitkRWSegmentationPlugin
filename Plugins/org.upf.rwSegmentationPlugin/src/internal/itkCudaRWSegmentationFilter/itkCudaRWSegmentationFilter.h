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

#ifndef itkCudaRWSegmentationFilter_h
#define itkCudaRWSegmentationFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"

#include "cuda_runtime.h"
#include "cublas.h"
#include "cusparse.h"

namespace itk
{
template <typename TInputImage, typename TOutputImage>
class CudaRWSegmentationFilter : public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef CudaRWSegmentationFilter Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaRWSegmentationFilter, ImageToImageFilter);

  /** Image type information. */
  typedef TInputImage InputImageType;
  typedef TOutputImage OutputImageType;

  itkSetMacro(LabelImage, typename OutputImageType::Pointer);
  itkGetMacro(LabelImage, typename OutputImageType::Pointer);

  itkSetMacro(Beta, double);
  itkGetMacro(Beta, double);

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
  CudaRWSegmentationFilter() : m_LabelImage(nullptr), m_Beta(1), m_Tolerance(1e-3),
                               m_MaximumNumberOfIterations(500), m_WriteBackground(true),
                               m_SolveForAllLabels(false), m_SolverIterations(0),
                               m_SolverError(1)  {}

  virtual ~CudaRWSegmentationFilter() {}
  void PrintSelf(std::ostream &os, Indent indent) const ITK_OVERRIDE;

  /** Does the actual work */
  void GenerateData() ITK_OVERRIDE;
  virtual int BiCGStab(int, int);
  virtual int AllocGPUMemory(int, int);
  virtual int FreeGPUMemory();

private:
  CudaRWSegmentationFilter(const Self &) ITK_DELETE_FUNCTION;
  void operator=(const Self &) ITK_DELETE_FUNCTION;

  typename OutputImageType::Pointer m_LabelImage;
  double m_Beta;
  double m_Tolerance;
  int m_MaximumNumberOfIterations;
  bool m_WriteBackground;
  bool m_SolveForAllLabels;

  int m_SolverIterations;
  float m_SolverError;

  float *cooValAdev, *cooValAdevM, *xdev, *r, *r_tld, *p, *p_hat, *s, *s_hat, *t, *v;
  float bnrm2, snrm2, error, alpha, beta, omega, rho, rho_1, resid;

  cudaError_t cudaStat1, /*cudaStat2,*/ cudaStat3, cudaStat4, cudaStat5, cudaStat6;
  cudaError_t cudaStat7, cudaStat8, cudaStat9, cudaStat10, cudaStat11, cudaStat12, cudaStat13;
  cudaError_t cudaStat14, /*cudaStat15,*/ cudaStat16, cudaStat17;
  cudaError_t cudaStat21, cudaStat22, cudaStat23, cudaStat24;

  cublasStatus cublas_status;

  float *probabilities, *probabilitiesInner;
  int *cooRowPtrAhost, /* *cooColPtrAhost, */ *csrColPtrAhost;
  int *cooRowPtrAhostM, /* *cooColPtrAhostM, */ *csrColPtrAhostM;
  float *cooValAhost, *cooValAhostM, *bhost, *xhost;
  int *cooRowPtrAdev, *cooColPtrAdev, *csrColPtrAdev;
  int *cooRowPtrAdevM, *cooColPtrAdevM, *csrColPtrAdevM;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCudaRWSegmentationFilter.hxx"
#endif

#endif