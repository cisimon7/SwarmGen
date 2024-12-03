import torch as th
from ..utils import weights_init


class MaskedConv2d(th.nn.Conv2d):
	MASK_TYPES = ["A", "B"]
	def __init__(self, mask_type, *args, **kwargs):
		super(MaskedConv2d, self).__init__(*args, **kwargs)
		assert mask_type in self.MASK_TYPES
		self.mask_type = mask_type
		
		self.register_buffer('mask', self.weight.data.clone())
		_, _, kH, kW = self.weight.size()

		self.mask.fill_(1)
		self.mask[:, :, kH // 2, kW // 2 + (mask_type == "B"):] = 0
		self.mask[:, :, kH // 2 + 1:] = 0
		
		self.apply(weights_init)
	
	def forward(self, x):
		self.weight.data *= self.mask
		return super(MaskedConv2d, self).forward(x)
	

class VerticalStackConv2d(MaskedConv2d):
	def __init__(self, mask_type, *args, **kwargs):
		super(VerticalStackConv2d, self).__init__(mask_type, *args, **kwargs)
		_, _, kH, kW = self.weight.size()
		self.mask.fill_(1)
		self.mask[:, :, kH//2:] = 0


class HorizontalStackConv2d(MaskedConv2d):
	def __init__(self, mask_type, *args, **kwargs):
		super(HorizontalStackConv2d, self).__init__(mask_type, *args, **kwargs)
		_, _, kH, kW = self.weight.size()
		self.mask.fill_(0)
		self.mask[:, :, kH//2, :kW // 2 + (mask_type == "B")] = 1


class AutoRegConv2d(th.nn.Module):
	def __init__(self, mask_type, *args, **kwargs):
		super(AutoRegConv2d, self).__init__()
		self.v_conv2d = VerticalStackConv2d(mask_type, *args, **kwargs)
		self.h_conv2d = HorizontalStackConv2d(mask_type, *args, **kwargs)
		
	def forward(self, x):
		vx = self.v_conv2d(x) 
		vy = self.h_conv2d(x)
		return vx + vy
	

class GatedMaskedConv2d(th.nn.Module):
	def __init__(self, *args, **kwargs):
		super(GatedMaskedConv2d, self).__init__()
		self.masked_conv_1 = MaskedConv2d(*args, **kwargs)
		self.masked_conv_2 = MaskedConv2d(*args, **kwargs)
		self.tanh = th.nn.Tanh()
		self.sigm = th.nn.Sigmoid()

	def forward(self, x):
		inp = self.tanh(self.masked_conv_1(x))
		gate = self.sigm(self.masked_conv_2(x))
		return inp*gate
	

class CondGatedMaskedConv2d(th.nn.Module):
	def __init__(self, *args, **kwargs):
		super(CondGatedMaskedConv2d, self).__init__()
		self.masked_conv_1 = AutoRegConv2d(*args[1:], **kwargs)
		self.masked_conv_2 = AutoRegConv2d(*args[1:], **kwargs)
		self.cond_conv_1 = th.nn.Conv2d(args[0], args[3], 1)
		self.cond_conv_2 = th.nn.Conv2d(args[0], args[3], 1)
		self.tanh = th.nn.Tanh()
		self.sigm = th.nn.Sigmoid()
		
		self.cond_conv_1.apply(weights_init)
		self.cond_conv_2.apply(weights_init)

	def forward(self, x, h):
		inp = self.tanh(self.masked_conv_1(x) + self.cond_conv_1(h))
		gate = self.sigm(self.masked_conv_2(x) + self.cond_conv_2(h))
		return inp * gate
		

class LabelNet(th.nn.Module):
	def __init__(self, num_agent, lheight, lwidth, nf, input_shape=10, output_shape=None):
		super(LabelNet, self).__init__()
		if output_shape is None:
			output_shape = [3, 3]
			
		self.input_shape = input_shape
		self.output_shape = output_shape
		# self.linear = th.nn.Linear(input_shape, th.prod(th.tensor(self.output_shape)))
		self.linear = th.nn.Sequential(
			th.nn.ConvTranspose2d(num_agent, nf*8, (2, 3), 1, bias=False),
			th.nn.BatchNorm2d(nf*8),
			th.nn.LeakyReLU(True),
			th.nn.Dropout2d(0.2),

			th.nn.ConvTranspose2d(nf*8, nf*4, 5, 1, bias=False), #8,6
			th.nn.BatchNorm2d(nf*4),
			th.nn.LeakyReLU(True),
			th.nn.Dropout2d(0.2),

			th.nn.ConvTranspose2d(nf*4, num_agent, 5, 1, bias=False), #10,8
			th.nn.BatchNorm2d(num_agent),
			th.nn.LeakyReLU(True),
			th.nn.Dropout2d(0.2),

			th.nn.AdaptiveAvgPool2d((lheight, lwidth)),
			th.nn.Linear(lwidth, lwidth),
		)

	def forward(self, h):
		return self.linear(h)#.view(-1, 1, *self.output_shape)


class LambdaModule(th.nn.Module):
	def __init__(self, fun):
		super().__init__()
		self.fun = fun

	def forward(self, x):
		return self.fun(x)


def get_conv2D(cond_in_channel, type, inp_chan, out_chan, inp_height, inp_width, bias=False):
	return CondGatedMaskedConv2d(
		cond_in_channel, type, inp_chan, out_chan, 
		(inp_height + 1 - (inp_height % 2), inp_width + 1 - (inp_width % 2)), 1, 
		(inp_height//2, inp_width//2), bias=bias
	)

class PixelCNN(th.nn.Module):
	def __init__(self, num_agent, lheight, lwidth, nf, n_channels=32, n_layers=7, emb_num=512, cond_input_shape=96, cond_output_shape=9):
		super(PixelCNN, self).__init__()

		self.label_net = LabelNet(num_agent, lheight, lwidth, nf, cond_input_shape, cond_output_shape)

		self.layers = th.nn.ModuleList()
		self.layers.append(th.nn.BatchNorm2d(num_agent))
		self.layers.append(
			CondGatedMaskedConv2d(num_agent, 'A', num_agent, n_channels, 7, 1, 3, bias=False)
			# get_conv2D(num_agent, 'A', num_agent, n_channels, lheight, lwidth, bias=False)
		)
		self.layers.append(th.nn.BatchNorm2d(n_channels))

		for _ in range(1, n_layers+1):
			self.layers.append(
				CondGatedMaskedConv2d(num_agent, 'B', n_channels, n_channels, 7, 1, 3, bias=False)
				# get_conv2D(num_agent, 'B', n_channels, n_channels, 7, 7, bias=False)
			)
			self.layers.append(th.nn.BatchNorm2d(n_channels))
			self.layers.append(th.nn.Dropout2d(0.1))

		self.layers.append(th.nn.Conv2d(n_channels, emb_num, 1))
		self.layers[-1].apply(weights_init)
		# self.layers.append(th.nn.Conv2d(
		# 	n_channels, emb_num, 
		# 	(lheight + 1 - (lheight % 2), lwidth + 1 - (lwidth % 2)), 1, 
		# 	(lheight//2, lwidth//2), bias=True
		# ))

	def forward(self, x, h):
		h = self.label_net(h)
		out = x
		for layer in self.layers:
			if isinstance(layer, CondGatedMaskedConv2d):
				out = layer(out, h)
			else:
				out = layer(out)
		return out