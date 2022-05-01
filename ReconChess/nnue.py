class NNUEReader():
  def __init__(self, f, feature_set):
    self.f = f
    self.feature_set = feature_set
    self.model = M.NNUE(feature_set)
    fc_hash = NNUEWriter.fc_hash(self.model)

    self.read_header(feature_set, fc_hash)
    self.read_int32(feature_set.hash ^ (M.L1*2)) # Feature transformer hash
    self.read_feature_transformer(self.model.input, self.model.num_psqt_buckets)
    for i in range(self.model.num_ls_buckets):
      l1 = nn.Linear(2*M.L1//2, M.L2+1)
      l2 = nn.Linear(M.L2, M.L3)
      output = nn.Linear(M.L3, 1)

      self.read_int32(fc_hash) # FC layers hash
      self.read_fc_layer(l1)
      self.read_fc_layer(l2)
      self.read_fc_layer(output, is_output=True)

      self.model.layer_stacks.l1.weight.data[i*(M.L2+1):(i+1)*(M.L2+1), :] = l1.weight
      self.model.layer_stacks.l1.bias.data[i*(M.L2+1):(i+1)*(M.L2+1)] = l1.bias
      self.model.layer_stacks.l2.weight.data[i*M.L3:(i+1)*M.L3, :] = l2.weight
      self.model.layer_stacks.l2.bias.data[i*M.L3:(i+1)*M.L3] = l2.bias
      self.model.layer_stacks.output.weight.data[i:(i+1), :] = output.weight
      self.model.layer_stacks.output.bias.data[i:(i+1)] = output.bias

  def read_header(self, feature_set, fc_hash):
    self.read_int32(VERSION) # version
    self.read_int32(fc_hash ^ feature_set.hash ^ (M.L1*2))
    desc_len = self.read_int32()
    description = self.f.read(desc_len)

  def tensor(self, dtype, shape):
    d = numpy.fromfile(self.f, dtype, reduce(operator.mul, shape, 1))
    d = torch.from_numpy(d.astype(numpy.float32))
    d = d.reshape(shape)
    return d

  def read_feature_transformer(self, layer, num_psqt_buckets):
    shape = layer.weight.shape

    bias = self.tensor(numpy.int16, [layer.bias.shape[0]-num_psqt_buckets]).divide(self.model.quantized_one)
    # weights stored as [num_features][outputs]
    weights = self.tensor(numpy.int16, [shape[0], shape[1]-num_psqt_buckets])
    weights = weights.divide(self.model.quantized_one)
    psqt_weights = self.tensor(numpy.int32, [shape[0], num_psqt_buckets])
    psqt_weights = psqt_weights.divide(self.model.nnue2score * self.model.weight_scale_out)

    layer.bias.data = torch.cat([bias, torch.tensor([0]*num_psqt_buckets)])
    layer.weight.data = torch.cat([weights, psqt_weights], dim=1)

  def read_fc_layer(self, layer, is_output=False):
    kWeightScale = self.model.weight_scale_out if is_output else self.model.weight_scale_hidden
    kBiasScale = self.model.weight_scale_out * self.model.nnue2score if is_output else self.model.weight_scale_hidden * self.model.quantized_one
    kMaxWeight = self.model.quantized_one / kWeightScale

    # FC inputs are padded to 32 elements by spec.
    non_padded_shape = layer.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    layer.bias.data = self.tensor(numpy.int32, layer.bias.shape).divide(kBiasScale)
    layer.weight.data = self.tensor(numpy.int8, padded_shape).divide(kWeightScale)

    # Strip padding.
    layer.weight.data = layer.weight.data[:non_padded_shape[0], :non_padded_shape[1]]

  def read_int32(self, expected=None):
    v = struct.unpack("<I", self.f.read(4))[0]
    if expected is not None and v != expected:
      raise Exception("Expected: %x, got %x" % (expected, v))
    return v
