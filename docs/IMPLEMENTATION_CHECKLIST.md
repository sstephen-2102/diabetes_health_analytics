# Implementation Checklist

- [x] Place actual CSV in `data/raw/` (downloaded and verified automatically; raw data stay out of Git).
- [x] Verify exact columns and target coding.
- [x] Verify Age, Sex, Education, Income mappings.
- [x] Freeze variable metadata.
- [x] Implement Member 1 modules.
- [x] Implement Member 2 modules (ANOVA, post-hoc and multiple-comparison adjustment scoped out by team decision).
- [x] Implement Member 3 modules.
- [x] Integrate service layer.
- [x] Build Streamlit pages.
- [x] Add tests.
- [ ] Validate report/presentation outputs.

## Final report checks (`docs/FINAL_REPORT.md`)

- [x] Course, names, date and app URL filled in.
- [x] Prose drafted for every section; guidance and source notes removed.
- [x] App screenshots in Appendix B (`docs/figures/app/`).
- [x] Every reference cited in the text and every citation listed (APA 7).
- [x] No "diagnosis" language except disclaimers; target wording consistent.
- [x] Deviation from the spec stated: no ANOVA, post-hoc tests or multiplicity adjustment (team decision).
- [x] All ten spec limitations present.
- [ ] Prose reviewed and rewritten by the team.
- [ ] Every figure and table caption still names its dataset variant after editing.
- [x] Final commit hash added (title page and Appendix A).
- [x] App opens without signing in (checked in a private window).
