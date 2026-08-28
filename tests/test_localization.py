import unittest, numpy as np
from asb_drx.localization import *

class LocalizationTests(unittest.TestCase):
    def setUp(self): self.c = LocalizationCriteria(0.4, 20.0, 0.1, 3.0, 3, 0.05)
    def test_homogeneous_field_is_not_localized(self):
        a,w=plastic_localization_geometry(np.ones((16,16)),1e-6); self.assertEqual(a,1.0); self.assertEqual(w,16e-6)
    def test_band_has_small_active_fraction_and_finite_width(self):
        r=np.zeros((16,16)); r[:,6:10]=1; a,w=plastic_localization_geometry(r,1e-6); self.assertEqual(a,0.25); self.assertAlmostEqual(w,4e-6)
    def test_all_criteria_and_persistence_are_required(self):
        good=LocalizationSnapshot(.25,4e-6,25,.2); bad=LocalizationSnapshot(.25,4e-6,5,.2)
        self.assertFalse(classify_localization((good,bad,good,good,good),1e-6,self.c).localized)
        d=classify_localization((bad,good,good,good),1e-6,self.c); self.assertTrue(d.localized); self.assertEqual(d.onset_index,1)
    def test_underresolved_band_is_rejected(self):
        d=classify_localization((LocalizationSnapshot(.2,2e-6,30,.2),)*4,1e-6,self.c); self.assertFalse(d.localized); self.assertIn("resolved_width",d.failed_criteria)
    def test_history_uses_matched_temperature_control_and_prior_peak(self):
        rates=np.ones((3,8,8)); rates[1:,:,3:5]=10; temp=np.full_like(rates,1000.); control=temp.copy(); temp[1:,:,3:5]+=25
        h=localization_history(rates,temp,control,np.array([100.,120.,90.]),1e-6); self.assertEqual(h[0].softening_fraction,0); self.assertAlmostEqual(h[2].softening_fraction,.25); self.assertEqual(h[2].temperature_excess_K,25)
    def test_refinement_requires_onset_and_width(self):
        self.assertTrue(refinement_passes(.2,.202,4e-6,4.1e-6,.05)); self.assertFalse(refinement_passes(.2,.25,4e-6,4.1e-6,.05))

if __name__ == '__main__': unittest.main()
