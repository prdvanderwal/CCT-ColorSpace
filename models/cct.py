from torch.hub import load_state_dict_from_url
import torch.nn as nn
from src.utils.transformers import TransformerClassifier
from src.utils.tokenizer import Tokenizer
from src.utils.helpers import pe_check, fc_check

try:
    from timm.models import register_model
except ImportError:
    from src.utils.registry import register_model

class CCT(nn.Module):
    def __init__(self,
                 img_size=32,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 extra_LN=False, # Customization
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            extra_LN=extra_LN,
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def _cct(arch, pretrained, progress, num_layers, num_heads, mlp_ratio, embedding_dim, extra_LN,
         kernel_size=3, stride=None, padding=None, positional_embedding='learnable',
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CCT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                extra_LN=extra_LN
                *args, **kwargs)
    return model

def cct_7(arch, pretrained, progress, extra_LN, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                extra_LN=extra_LN, *args, **kwargs) # Extra LN customization


################################# Added for DL #########################################
@register_model
def CCT_7(pretrained=False, progress=False,
          img_size=32, positional_embedding='learnable', num_classes=10, extra_LN=True, *args, **kwargs):
    return cct_7('CCT_7', pretrained, progress, extra_LN=extra_LN,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes, *args, **kwargs)
#################################  Until here  #########################################


