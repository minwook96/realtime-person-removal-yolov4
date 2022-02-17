####################################################################################
#Start Of unused functions -- used to fix the color difference problem but not work#
####################################################################################
def convert_color(source, target):
	def image_stats(image):
		# compute the mean and standard deviation of each channel
		l, a, b = cv2.split(image)
		l_mean, l_std = (l.mean(), l.std())
		a_mean, a_std = (a.mean(), a.std())
		b_mean, b_std = (b.mean(), b.std())
		# return the color statistics
		return l_mean, l_std, a_mean, a_std, b_mean, b_std

	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
	# subtract the means from the target image
	l, a, b = cv2.split(target)
	l -= lMeanTar.astype(np.uint8)
	a -= aMeanTar.astype(np.uint8)
	b -= bMeanTar.astype(np.uint8)
	# scale by the standard deviations
	l = (lStdTar / lStdSrc) * l
	a = (aStdTar / aStdSrc) * a
	b = (bStdTar / bStdSrc) * b
	# add in the source mean
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc
	# clip the pixel intensities to [0, 255] if they fall outside
	# this range
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)
	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

	# return the color transferred image
	return transfer


def hist_equal(image):
	hist, bins = np.histogram(image.flatten(), 256, [0, 256])
	cdf = hist.cumsum()
	cdf_m = np.ma.masked_equal(cdf, 0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m, 0).astype('uint8')
	send_image = cdf[image]
	return send_image


def convert_color_linear(target, ref_frame, results):
	def is_in_res(point_x, point_y, reses):
		for res in reses:
			if res[0] < x < res[0] + res[2] and res[1] < y < res[1] + res[3]:
				return True
		return False

	def solve_equation(A, B):
		from itertools import combinations
		a = np.matrix(A)
		b = np.matrix(B)
		num_vars = a.shape[1]
		rank = np.linalg.matrix_rank(a)
		if rank == num_vars:
			sol = np.linalg.lstsq(a, b)[0]  # not under-determined
		elif False:
			for nz in combinations(range(num_vars), rank):  # the variables not set to zero
				try:
					sol = np.zeros((num_vars, 1))
					sol[nz, :] = np.asarray(np.linalg.solve(a[:, nz], b))
				except np.linalg.LinAlgError:
					pass
		return np.array(sol)

	_h, _w, _c = ref_frame.shape
	res = []
	for i in results:
		w = int(i[1][2]*_w*1.2)
		h = int(i[1][3]*_h*1.2)
		x = int(i[1][0]*_w - w/2)
		y = int(i[1][1]*_h - h/2)
		res.append([x, y, w, h])
	random_points = []
	count = 0
	while len(random_points) < 200:
		# get random point
		x = random.randrange(0, _w)
		y = random.randrange(0, _h)
		count += 1
		if count > 5000:
			break
		if not is_in_res(x, y, res):
			random_points.append([x, y])
	showimg = target.copy()
	for x, y in random_points:
		showimg = cv2.circle(showimg, (x, y), 2, (0, 0, 255), -1)
	cv2.imshow('qweqwe', showimg)

	lab_target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
	lab_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2LAB)
	target_colors = []
	ref_colors = []
	for x, y in random_points:
		temp_lab = lab_target[y][x]
		target_colors.append([temp_lab[0], temp_lab[1], temp_lab[2]])
		temp_lab = lab_ref[y][x]
		ref_colors.append([temp_lab[0], temp_lab[1], temp_lab[2]])
		# ref_color = target_color*M
	M = solve_equation(ref_colors, target_colors)
	result_image = np.rint(lab_target.dot(M)).astype('uint8')

	result = cv2.cvtColor(result_image, cv2.COLOR_LAB2BGR)

	cv2.imshow('Before', target)
	cv2.imshow('After', result)
	return result


def hist_match(source, template):
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()

	# get the set of unique pixel values and their corresponding indices and
	# counts
	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)

	# take the cumsum of the counts and normalize by the number of pixels to
	# get the empirical cumulative distribution functions for the source and
	# template images (maps pixel value --> quantile)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]

	# interpolate linearly to find the pixel values in the template image
	# that correspond most closely to the quantiles in the source image
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

	return interp_t_values[bin_idx].reshape(oldshape)
##################################################################################
#End Of unused functions -- used to fix the color difference problem but not work#
##################################################################################
