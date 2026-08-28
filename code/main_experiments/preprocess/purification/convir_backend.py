from copy import copy


class Purify:
    def __init__(self, args, config, type, dataset):
        from preprocess.convir.purifier import Purify as ConvIRPurify

        convir_args = copy(args)
        convir_args.restorer_ckpt = getattr(args, "restorer_ckpt", "")
        convir_args.restorer_version = getattr(args, "restorer_version", "base")
        convir_args.use_conv_ir = not getattr(args, "disable_conv_ir", False)
        self.impl = ConvIRPurify(convir_args, config, type, dataset)

    def pur(self):
        return self.impl.pur()


__all__ = ["Purify"]
