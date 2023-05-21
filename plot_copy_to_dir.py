import os
import shutil

#datadir='/home/ubuntu/testing/'
#webdir='/home/ubuntu/testing/web/v1/data'
#datadir='/Users/mbar0087/Documents/Monash Weather Web/'
datadir='/g/data/w40/mb0427/MonashWeb/forecasts/'
webdir='/home/565/mb0427/MonashWeb/monash_weather_web/data/'
#webdir='/Users/mbar0087/Documents/Monash Weather Web/v1/data'

model='GFS'
domains=['IndianOcean','Australia','PacificOcean','SouthAmerica',
          'SouthernAfrica','SH']

plottypes=['UpperLevel',
            'Precip6H',
            'IVT',
            'IPV-320K',
            'IPV-330K',
            'IPV-350K',
            'IrrotPV']

indir=datadir#+'/'+model+'/'+'forecasts/'

files = os.listdir(indir)
for type in plottypes:
    for domain in domains:
        outdir=webdir+'/'+model+'/'+domain+'/'+type+'/'
        isExist = os.path.exists(outdir)
        if not isExist:
            os.makedirs(outdir)
        for file in files:
            full_file_name = os.path.join(indir, file)
            if file.startswith(model+"_"+domain+"_"+type+"_"):
                shutil.copy(full_file_name, outdir)
                os.rename(os.path.join(outdir, file), os.path.join(outdir, file).replace(domain+'_', ''))
