import sys
from fmriprep_helpers import *

# path to the fmriprep, freesurfer and hcp folders
config.DATADIR = 'output'

# path to source code
config.sourceDir = '.'

# output folder
config.outDir = 'rsDenoise'

# use Cifti files?
config.isCifti = True
config.space = 'MNI152NLin2009cAsym_res-2'

# use sge?
config.queue = True
config.sgeopts  = '-pe threaded 3-3 -q all.q'

# Uncomment to specify fmriFileTemplate
#config.fmriFileTemplate        = '#fMRIsession#_#fMRIrun#.nii.gz'
#config.fmriFileTemplate        = '#fMRIsession#_#fMRIrun#_Atlas_s0.dtseries.nii'

# Choose pipeline
config.pipelineName = 'A'
config.interpolation = 'linear'
config.Operations   = config.operationDict[config.pipelineName]

# parcellation for FC matrix
config.parcellationName = 'Schaefer'
config.parcellationFile = 'Schaefer2018_400Parcels_7Networks_order.dlabel.nii'
config.nParcels         = 400

# overwrite previous files?
config.overwrite = True

fmriRuns = ['task-rest_run-01']
subjects = ['sub-0040']

keepSub = np.zeros((len(subjects)),dtype=np.bool_)
iSub=0
for config.subject in subjects:
    iRun = 0
    for config.fmriRun in fmriRuns:
        if not config.queue:
            print('SUB {}/{} [{}]: run {}/{} [{}]'.format(iSub+1,len(subjects),config.subject,iRun+1,len(fmriRuns),config.fmriRun))
        keepSub[iSub] = runPipelinePar(launchSubproc=False)
        if not keepSub[iSub]:
            break
        iRun+=1
    iSub+=1
print('Keeping {}/{} subjects'.format(np.sum(keepSub),len(subjects)))

# launch array job (if there is something to do)
if len(config.scriptlist)>0:
    fnSubmitJobArrayFromJobList()

checkProgress(pause=60,verbose=False)