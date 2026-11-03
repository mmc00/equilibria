All the files needed to run the GAMS version of the GTAP model are contained in
the zip file. They can all be copied to the same folder. The article
contains an extensive section on how to run the model. This short 'README'
file contains an abbreviated outline of the data and model files.

The database files, i.e. the 10 x 10 aggregation of the GTAP Data Base are saved
as HAR-formatted files using the GTAPAgg2 program. They include:

   basedata-10x10.har   The base data
   sets-10x10.har       The sets file
   default-10x10.har    The default parameter file

These need to be converted to the 'GDX' format. The utility 'HAR2GDX.EXE'
converts from HAR format to GDX format and is available at:
http://www.copsmodels.com/gp-gams.htm. Users can convert the files one
by one, or use the included 'convert.cmd' file that automates the
conversion of all needed files. The user needs to change the 'BaseName'
on line three of the 'convert.cmd' batch file. It can be invoked
by typing 'convert' in a Windows terminal screen.
(N.B. The zip file also includes a 9 x 10 aggregation that illustrates
the potential use of a non-diagonal 'make' matrix.)

The core model files are:

   model.gms
   getData.gms
   cal.gms
   postsim.gms
   iterloop.gms
   solve.gms
   mvar.gms

These files normally do not require any modification by the user.

The zip file contains 6 simulation files:

   comp.gms       Model diagnostic--residual and homogeneity checks.
   gft.gms        Global free trade with standard specification.
   gftCapFix.gms  Global free trade with exogenous capital account.
   gftCET.gms     Global free trade with CET implementation for domestic output.
   GFTLnd.gms     Global free trade with upward sloping land supply curves.
   GFTFlxFF.gms   Global free trade with upward sloping nat. res. supply curves.

Any new version of the model, i.e. changes to model specification, key
parameters or aggregation, should be tested with a run of 'comp.gms' to make
sure that model initialization and calibration are done consistently and
that the model passes the homogeneity test (unless a model change introduces
a price or nominal rigidity.) The simulation files are self-contained, i.e.
they can be run by simply invoking the GAMS program, for example:

   gams comp

They can also be invoked from within the GAMS IDE. The simulation files
will read in the various 'include' files that fully define a simulation.

The supplementary materials include three additional files. The first is
called 'runall.cmd'. This file automates the running of all global
free trade simulations. Just type 'runall' in a Windows terminal screen to
invoke the batch command. The last line of 'runall.cmd' invokes a second
batch file called 'merge.cmd'. This file facilitates the creation of a
CSV cube that merges the results from the various free trade simulations.
It can readily be modified by the user to include a different set of
simulations. The 'merge.cmd' file invokes a separate GAMS program called
'merge.gms' that extracts results from a GDX file and adds them to a user-
defined CSV cube.

In summary, users need the following software:

   GTAPAGG2       (Only needed to have own aggregation of the GTAP Data Base.)
   HAR2GDX.EXE
   GAMS
