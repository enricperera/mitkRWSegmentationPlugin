project(org_upf_rwSegmentationPlugin)

mitk_create_plugin(
  EXPORT_DIRECTIVE RWSEGMENTATIONPLUGIN_EXPORT
  EXPORTED_INCLUDE_SUFFIXES src
  MODULE_DEPENDS MitkQtWidgetsExt
)
