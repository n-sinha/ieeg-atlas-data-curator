import numpy as np

def merge_rois(customAAL, roiAAL, atlas):
    
    # Convert inputs into appropriate types (assuming customAAL and roiAAL are Pandas DataFrames)
    for roi in range(len(customAAL)):
        # Assign parcel1 from roiAAL based on customAAL.Roi1
        customAAL.loc[roi, 'parcel1'] = roiAAL.loc[customAAL.loc[roi, 'Roi1'], 'parcel']
        
        # Handle Roi2 assignment
        if np.isnan(customAAL.loc[roi, 'Roi2']):
            customAAL.loc[roi, 'parcel2'] = np.nan
        else:
            customAAL.loc[roi, 'parcel2'] = roiAAL.loc[customAAL.loc[roi, 'Roi2'], 'parcel']
            atlas['data'][atlas['data'] == customAAL.loc[roi, 'parcel2']] = customAAL.loc[roi, 'parcel1']
        
        # Handle Roi3 assignment
        if np.isnan(customAAL.loc[roi, 'Roi3']):
            customAAL.loc[roi, 'parcel3'] = np.nan
        else:
            customAAL.loc[roi, 'parcel3'] = roiAAL.loc[customAAL.loc[roi, 'Roi3'], 'parcel']
            atlas['data'][atlas['data'] == customAAL.loc[roi, 'parcel3']] = customAAL.loc[roi, 'parcel1']
    
    # Handle inclusion/exclusion logic
    included = np.concatenate((customAAL['Roi1'], customAAL['Roi2'], customAAL['Roi3']))
    included = included[~np.isnan(included)]  # Remove NaN values
    
    excluded = np.setxor1d(roiAAL['Sno'], included)
    atlas['data'][np.isin(atlas['data'], roiAAL['parcel'][excluded])] = 0

    # Create atlasCustom and roiAALcustom as outputs
    atlasCustom = atlas
    
    roiAALcustom = {}
    roiAALcustom['Sno'] = np.arange(1, len(customAAL) + 1)
    roiAALcustom['Regions'] = customAAL['Roi_name']
    roiAALcustom['Lobes'] = customAAL['Lobes']
    roiAALcustom['isSideLeft'] = customAAL['Roi_name'].str.endswith('_L')
    roiAALcustom['parcel'] = customAAL['parcel1']
    
    # Calculate coordinates (CRS to RAS transformation)
    xyz = []
    for roi in range(len(customAAL)):
        indices = np.argwhere(atlas['data'] == roiAALcustom['parcel'][roi])
        CRS = np.hstack([indices, np.full((indices.shape[0], 1), roiAALcustom['parcel'][roi])])
        
        RAS = np.dot(atlas['hdr']['Transform']['T'].T, np.hstack([CRS[:, :3], np.ones((CRS.shape[0], 1))]).T).T
        RAS = RAS[:, :3]
        xyz.append(RAS.mean(axis=0))
    xyz = np.array(xyz)
    
    roiAALcustom['x'] = xyz[:, 0]
    roiAALcustom['y'] = xyz[:, 1]
    roiAALcustom['z'] = xyz[:, 2]

    # Convert roiAALcustom to Pandas DataFrame for easier use
    roiAALcustom = pd.DataFrame(roiAALcustom)
    
    return atlasCustom, roiAALcustom