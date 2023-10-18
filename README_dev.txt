While developing at NSIDC, the workflow is:

Create a dev VM with the `seaice_ecdr_vm` project
(https://bitbucket.org/nsidc/seaice_ecdr_vm)

`pm_icecon`, `seaice_ecdr`, and `pm_tb_data` will be checked out to
`/home/vagrant/{project_name}`.

in ~/seaice_ecdr/:

   Can run unit and integration tests -- or manual subsets of -- with:
     ./scripts/run_ecdr_pytest.sh

   Create an Initial Daily ECDR file with e.g.,:

   ./scripts/cli.sh idecdr --date 2021-04-05 --hemisphere north --resolution 12 --output-dir /tmp/
