
README_dev.txt

While developing, the workflow is:

pm_icecon:
   Download pm_icecon repo to ~/pm_icecon
   run 'python setup.py develop'
   Current branch: origin/update_for_nise_cdr_cetb

seaice_ecdr:
   Download seaice_ecdr repo to ~/seaice_ecdr
   run 'python setup.py develop'
   Current branch: origin/initial_pmicecon_ecdr_gen

in ~/seaice_ecdr/:

   Can run unit and integration tests -- or manual subsets of -- with:
     ./run_ecdr_pytest.sh

   Can generate a sample netCDF file with:
     python seaice_ecdr/tests/integration/gen_ide_sample.py

