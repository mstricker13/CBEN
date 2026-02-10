from esa_snappy import ProductIO
from esa_snappy import HashMap
from esa_snappy import GPF
import sys

# some notes regarding sar processing
# https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel1.html
# https://github.com/corteva/rioxarray/issues/339
# https://github.com/corteva/rioxarray/issues/376 
# https://www.google.com/search?q=python+how+to+orthorectify+sentinel+1+from+gcp&rlz=1C1TKQJ_enJP1116JP1116&oq=python+how+to+orthorectify+sentinel+1+from+gcp&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigAdIBCTI2MTcyajBqN6gCALACAA&sourceid=chrome&ie=UTF-8
# https://forum.sentinel-hub.com/t/sentinel1-orthorectification-using-srtm-or-aster-dem/3216

def extract_bands(source, bandnames):
    #print("\tExtracting Bands")
    parameters = HashMap()
    parameters.put('sourceBandNames', bandnames)
    output = GPF.createProduct('BandsExtractorOp', parameters, source)
    return output

def do_apply_orbit_file(source):
    #print('\tApply orbit file...')
    parameters = HashMap()
    parameters.put('Apply-Orbit-File', True)
    output = GPF.createProduct('Apply-Orbit-File', parameters, source)
    return output

def do_border_noise_removal(source, pols):
    #print('\tBorder noise removal...')
    parameters = HashMap()
    parameters.put('selectedPolarisations', pols)
    output = GPF.createProduct('Remove-GRD-Border-Noise', parameters, source)
    return output

def LinearToFromdB(source):
    #print('\tConverting coefficients...')
    parameters = HashMap()
    parameters.put('sourceBands', 'Sigma0_VH,Sigma0_VV')
    output = GPF.createProduct('LinearToFromdB', parameters, source)
    return output

def do_thermal_noise_removal(source):
    #print('\tThermal noise removal...')
    parameters = HashMap()
    parameters.put('removeThermalNoise', True)
    output = GPF.createProduct('ThermalNoiseRemoval', parameters, source)
    return output

def do_calibration(source, polarization, pols, outputImageScaleInDb):
    #print('\tCalibration...')
    parameters = HashMap()
    parameters.put('outputSigmaBand', True)
    if polarization == 'DH':
        parameters.put('sourceBands', 'Intensity_HH,Intensity_HV')
    elif polarization == 'DV':
        parameters.put('sourceBands', 'Intensity_VH,Intensity_VV')
    elif polarization == 'SH' or polarization == 'HH':
        parameters.put('sourceBands', 'Intensity_HH')
    elif polarization == 'SV':
        parameters.put('sourceBands', 'Intensity_VV')
    else:
        print("different polarization!")
    parameters.put('selectedPolarisations', pols)
    parameters.put('outputImageScaleInDb', outputImageScaleInDb)
    output = GPF.createProduct("Calibration", parameters, source)
    return output

def do_speckle_filtering(source):
    #print('\tSpeckle filtering...')
    parameters = HashMap()
    parameters.put('filter', 'Lee')
    #parameters.put('filterSizeX', 5)
    #parameters.put('filterSizeY', 5)
    output = GPF.createProduct('Speckle-Filter', parameters, source)
    return output

#def do_terrain_correction(source, proj, downsample):
def do_terrain_correction(source, downsample):
    #print('\tTerrain correction...')
    parameters = HashMap()
    #parameters.put('demName', 'GETASSE30')
    #parameters.put('demName', 'SRTM 3Sec')
    parameters.put('demName', 'Copernicus 30m Global DEM')
    #parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('imgResamplingMethod', 'CUBIC_CONVOLUTION')
    parameters.put('DEMResamplingMethod', 'CUBIC_CONVOLUTION')

    #proj = '''PROJCS["UTM Zone 4 / World Geodetic System 1984",GEOGCS["World Geodetic System 1984",DATUM["World Geodetic System 1984",SPHEROID["WGS 84", 6378137.0, 298.257223563, AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG","8901"]],UNIT["degree", 0.017453292519943295],AXIS["Geodetic longitude", EAST],AXIS["Geodetic latitude", NORTH]],PROJECTION["Transverse_Mercator"],PARAMETER["central_meridian", -159.0],PARAMETER["latitude_of_origin", 0.0],PARAMETER["scale_factor", 0.9996],PARAMETER["false_easting", 500000.0],PARAMETER["false_northing", 0.0],UNIT["m", 1.0],AXIS["Easting", EAST],AXIS["Northing", NORTH]]'''
    #parameters.put('mapProjection', proj)       # comment this line if no need to convert to UTM/WGS84, default is WGS84
    
    parameters.put('saveProjectedLocalIncidenceAngle', True)
    parameters.put('saveSelectedSourceBand', True)
    while downsample == 1:                      # downsample: 1 -- need downsample to 40m, 0 -- no need to downsample
        parameters.put('pixelSpacingInMeter', 40.0)
        break
    output = GPF.createProduct('Terrain-Correction', parameters, source)
    return output

def do_subset(source, wkt):
    #print('\tSubsetting...')
    parameters = HashMap()
    parameters.put('geoRegion', wkt)
    output = GPF.createProduct('Subset', parameters, source)
    return output
