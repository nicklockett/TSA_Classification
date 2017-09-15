import constants

# show the threat zones
#body_zones_img = plt.imread(BODY_ZONES)
#fig, ax = plt.subplots(figsize=(15,15))
#ax.imshow(body_zones_img)

# unit test -----------------------
#df = self.get_hit_rate_stats(THREAT_LABELS)
#df.head()

#------------------------------------------------------------------------------------------
# self.chart_hit_rate_stats(df_summary): charts threat probabilities in desc order by zone
#
# df_summary:                 a dataframe like that returned from self.get_hit_rate_stats(...)
#
#-------------------------------------------------------------------------------------------


# unit test ------------------
#self.chart_hit_rate_stats(df)


#------------------------------------------------------------------------------------------
# self.print_hit_rate_stats(df_summary): lists threat probabilities by zone
#
# df_summary:               a dataframe like that returned from self.get_hit_rate_stats(...)
#
#------------------------------------------------------------------------------------------


# unit test -----------------------
#self.print_hit_rate_stats(df)

#----------------------------------------------------------------------------------------
# def get_subject_labels(infile, subject_id): lists threat probabilities by zone
#
# infile:                          labels csv file
#
# subject_id:                      the individual you want the threat zone labels for
#
# returns:                         a df with the list of zones and contraband (0 or 1)
#
#---------------------------------------------------------------------------------------



    
# unit test ----------------------------------------------------------------------
#print(get_subject_labels(THREAT_LABELS, '00360f79fd6e02781457eda48f85da90'))


#----------------------------------------------------------------------------------
# self.get_single_image(infile, nth_image):  returns the nth image from the image stack
#
# infile:                              an aps file
#
# returns:                             an image
#----------------------------------------------------------------------------------



  

# unit test ---------------------------------------------------------------
#an_img = self.get_single_image(APS_FILE_NAME, 0)

#fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

#axarr[0].imshow(an_img, cmap=COLORMAP)
#plt.subplot(122)
#plt.hist(an_img.flatten(), bins=256, color='c')
#plt.xlabel("Raw Scan Pixel Value")
#plt.ylabel("Frequency")
#plt.show()

#----------------------------------------------------------------------------------
# self.convert_to_grayscale(img):           converts a ATI scan to grayscale
#
# infile:                              an aps file
#
# returns:                             an image
#----------------------------------------------------------------------------------


# unit test ------------------------------------------
#img_rescaled = self.convert_to_grayscale(an_img)

#fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

#axarr[0].imshow(img_rescaled, cmap=COLORMAP)
#plt.subplot(122)
#plt.hist(img_rescaled.flatten(), bins=256, color='c')
#plt.xlabel("Grayscale Pixel Value")
#plt.ylabel("Frequency")
#plt.show()

#-------------------------------------------------------------------------------
# self.spread_spectrum(img):        applies a histogram equalization transformation
#
# img:                         a single scan
#
# returns:                     a transformed scan
#-------------------------------------------------------------------------------


  
# unit test ------------------------------------------
#img_high_contrast = self.spread_spectrum(img_rescaled)

#fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

#axarr[0].imshow(img_high_contrast, cmap=COLORMAP)
#plt.subplot(122)
#plt.hist(img_high_contrast.flatten(), bins=256, color='c')
#plt.xlabel("Grayscale Pixel Value")
#plt.ylabel("Frequency")
#plt.show()

#-----------------------------------------------------------------------------------------
# self.roi(img, vertices):              uses vertices to mask the image
#
# img:                             the image to be masked
#
# vertices:                        a set of vertices that define the region of interest
#
# returns:                         a masked image
#-----------------------------------------------------------------------------------------

  
# unit test -----------------------------------------------------------------
#fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
#    
#i = 0
#for row in range(4):
#    for col in range(4):
#        an_img = self.get_single_image(APS_FILE_NAME, i)
#        img_rescaled = self.convert_to_grayscale(an_img)
#        img_high_contrast = self.spread_spectrum(img_rescaled)
#        if zone_slice_list[0][i] is not None:
#            masked_img = self.roi(img_high_contrast, zone_slice_list[0][i])
#            resized_img = cv2.resize(masked_img, (0,0), fx=0.1, fy=0.1)
#            axarr[row, col].imshow(resized_img, cmap=COLORMAP)
#        i += 1

#-----------------------------------------------------------------------------------------
# self.crop(img, crop_list):                
#
# img:                                 the image to be cropped
#
# crop_list:                           a crop_list entry with [x , y, width, height]
#
# returns:                             a cropped image
#-----------------------------------------------------------------------------------------

  
# unit test -----------------------------------------------------------------

#fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
#    
#i = 0
#for row in range(4):
#    for col in range(4):
#        an_img = self.get_single_image(APS_FILE_NAME, i)
#        img_rescaled = self.convert_to_grayscale(an_img)
#        img_high_contrast = self.spread_spectrum(img_rescaled)
#        if zone_slice_list[0][i] is not None:
#            masked_img = self.roi(img_high_contrast, zone_slice_list[0][i])
#            cropped_img = self.crop(masked_img, zone_crop_list[0][i])
#            resized_img = cv2.resize(cropped_img, (0,0), fx=0.1, fy=0.1)
#            axarr[row, col].imshow(resized_img, cmap=COLORMAP)
#        i += 1

#------------------------------------------------------------------------------------------
# self.normalize(image): 
#
# parameters:      image - a tsa scan
#
# returns:         a normalized image
#
#------------------------------------------------------------------------------------------


#unit test ---------------------
#an_img = self.get_single_image(APS_FILE_NAME, 0)
#img_rescaled = self.convert_to_grayscale(an_img)
#img_high_contrast = self.spread_spectrum(img_rescaled)
#masked_img = self.roi(img_high_contrast, zone_slice_list[0][0])
#cropped_img = self.crop(masked_img, zone_crop_list[0][0])
#normalized_img = self.normalize(cropped_img)
#print ('Normalized: length:width -> {:d}:{:d}|mean={:f}'.format(len(normalized_img), len(normalized_img[0]), normalized_img.mean()))
#print (' -> type ', type(normalized_img))
#print (' -> shape', normalized_img.shape)

#-------------------------------------------------------------------------------------
# self.zero_center(image): Shift normalized image data and move the range so it is 0 c
#                     entered at the PIXEL_MEAN
#
# parameters:         image
#
# returns:            a zero centered image
#
#-----------------------------------------------------------------------------------------------------------

#unit test ---------------------
#an_img = self.get_single_image(APS_FILE_NAME, 0)
#img_rescaled = self.convert_to_grayscale(an_img)
#img_high_contrast = self.spread_spectrum(img_rescaled)
#masked_img = self.roi(img_high_contrast, zone_slice_list[0][0])
#cropped_img = self.crop(masked_img, zone_crop_list[0][0])
#normalized_img = self.normalize(cropped_img)
#zero_centered_img = self.zero_center(normalized_img)
#print ('Zero Centered: length:width -> {:d}:{:d}|mean={:f}'.format(len(zero_centered_img), len(zero_centered_img[0]), zero_centered_img.mean()))
#print ('Conformed: Type ->', type(zero_centered_img), 'Shape ->', zero_centered_img.shape)


