TODO: this README out of date.

While developing, the workflow is:

pm_icecon:
   Download pm_icecon repo to ~/pm_icecon
   run 'python setup.py develop'

seaice_ecdr:
   Download seaice_ecdr repo to ~/seaice_ecdr
   run 'python setup.py develop'

in ~/seaice_ecdr/:

   Can run unit and integration tests -- or manual subsets of -- with:
     ./run_ecdr_pytest.sh
