import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from vit import Transformer


class MAE(nn.Module):
    '''
    the implementation from https://github.com/lucidrains/vit-pytorch.
    '''
    def __init__(self,
                 *,
                 encoder,
                 decoder_dim,
                 masking_ratio=0.75,
                 decoder_depth=1,
                 decoder_heads=8,
                 decoder_dim_head=64,
                 device='cpu'):
        super().__init__()
        # common
        self.device =  device
        assert 0 <= masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        # num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        num_patches, encoder_dim = encoder.pos_embed.shape[-2:]

        # self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim,
                                   depth=decoder_depth,
                                   heads=decoder_heads,
                                   dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    
    
    def forward_k(self,img,model_pre):
        '''
        input:
            img: [batchsize,3,h,w]
            model_pre = MAE-(ki-1)
        output:
            recon_tokens_loss = F.mse_loss(pred_tokens_values,tokens_pre)
        '''
        '''
        ???????????????
            tokens_pre????????????????????????MAE???ViT(encoder)???????????????->???????????????????????????
            model??????????????????->?????????????????????????????????->???????????????
            model?????????????????? forward(),???????????????forward_k()
            ?????????????????????MAE-K,??????????????????forward(),??????(K-1)?????????forward_k()
            ????????????K???,????????????,?????????????????????model.ckpt->????????????->model-K.ckpt
            ???????????????to_pixel???,?????????tokens?????????????????????decoder???to_pixel????????????tokens_pre??????????????????
        '''
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # prepare tokens_pre
        # preparing for tokens_pre,generating by model_pre with no grad
        model_pre.eval()
        with torch.no_grad():
            # mask ratio = 0 deit encoder output
            _,_,tokens_pre = model_pre.encoder.forward_encoder(img)
            
        # preparing tokens now
        # patch to encoder tokens and add positions
        tokens = self.encoder.patch_embed(img)
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=self.device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=self.device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        # append cls and dist token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, 0]
        dist_token = self.encoder.dist_token + self.encoder.pos_embed[:, 1]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)  
        dist_tokens = dist_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, dist_tokens, tokens), dim=1)
        tokens = self.encoder.pos_drop(tokens)
        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        tokens = self.encoder.norm(tokens)
        encoded_tokens = tokens
        # get the patches(now turn to tokens_pre) to be masked for the final reconstruction loss
        masked_patches = tokens_pre[batch_range, masked_indices]

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)
        # splice out the mask tokens and project to pixel values(now tokens_pre valuse)
        mask_tokens = decoded_tokens[:, :num_masked]
        pred_tokens_values = self.to_pixels(mask_tokens)

        recon_tokens_loss = F.mse_loss(pred_tokens_values, masked_patches)

        return recon_tokens_loss


    def forward(self, img):
        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=self.device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=self.device)[:, None]
        
        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # get tokens encoded by DeiT
        tokens = self.encoder.patch_embed(img)
        tokens = tokens + self.encoder.pos_embed[:, 2:(num_patches + 2)]
        # masking
        tokens = tokens[batch_range, unmasked_indices]
        # append cls and dist token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, 0]
        dist_token = self.encoder.dist_token + self.encoder.pos_embed[:, 1]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)  
        dist_tokens = dist_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, dist_tokens, tokens), dim=1)
        tokens = self.encoder.pos_drop(tokens)
        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        tokens = self.encoder.norm(tokens)

        encoded_tokens = tokens
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        
        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        
        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)
        
        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.to_pixels(mask_tokens)
        
        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss


class EvalNet(nn.Module):
    '''
    the encoder of masked auto-encoder + linear layer.
    '''
    def __init__(self, encoder, n_class, masking_ratio=0, device='cpu'):
        super(EvalNet, self).__init__()
        # common
        self.device = device
        assert 0 <= masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embed.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]

        # linear layer
        self.fc = nn.Linear((num_patches) * encoder_dim, n_class)

    def forward(self, img):
        # get patches
        patches = self.to_patch(img)
        
        batch, num_patches, *_ = patches.shape
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=self.device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        batch_range = torch.arange(batch, device=self.device)[:, None]

        # patch to encoder tokens and add positions
        tokens = self.encoder.patch_embed(img)
        # get the unmasked tokens to be encoded
        tokens = tokens[batch_range, unmasked_indices]
        # append cls and dist token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, 0]
        dist_token = self.encoder.dist_token + self.encoder.pos_embed[:, 1]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)  
        dist_tokens = dist_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, dist_tokens, tokens), dim=1)
        tokens = self.encoder.pos_drop(tokens)
        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        tokens = self.encoder.norm(tokens)
        encoded_tokens = tokens
        # feed to linear probing
        latent_fea = encoded_tokens.flatten(start_dim=1)
        output = self.fc(latent_fea)

        return output


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing from https://github.com/NVIDIA/DeepLearningExamples.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()