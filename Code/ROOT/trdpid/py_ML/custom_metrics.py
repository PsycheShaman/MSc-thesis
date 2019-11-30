import tensorflow as tf

# This is a Keras metric which you can put into your compile. e_eff in (0, 1.0). Just ignore thresh, only exists for numerical consistency of float comparisons.
def PionEfficiencyAtElectronEfficiency(e_eff, thresh = 1e-4):
	def PionEfficiencyAtElectronEfficiency(y_true, y_pred):
		e_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 1.0), thresh))
		p_pred = tf.boolean_mask(y_pred, tf.less(tf.abs(y_true - 0.0), thresh))
		argsort = tf.argsort(e_pred)
		e_eff_90_cutoff = e_pred[argsort[tf.cast(tf.multiply(tf.cast(tf.shape(argsort)[-1], dtype='float32'), (1 - e_eff)), dtype='int32')]]
		return tf.cast(tf.count_nonzero(p_pred > e_eff_90_cutoff) / tf.count_nonzero(tf.equal(y_true, 0)), dtype='float32')
	return PionEfficiencyAtElectronEfficiency