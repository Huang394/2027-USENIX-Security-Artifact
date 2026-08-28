def build_purifier(args, config, type, dataset):
    backend = getattr(args, "purifier_backend", "zip")
    if backend == "zip":
        from .zip_backend import Purify
    elif backend == "var_deg_zip":
        from .var_deg_zip_backend import Purify
    elif backend == "diffusionzip":
        from .diffusionzip_backend import Purify
    elif backend == "convir_zip":
        from .convir_backend import Purify
    elif backend == "sampdetox":
        from .sampdetox_backend import Purify
    elif backend == "none":
        from .none_backend import Purify
    else:
        raise ValueError(f"Unsupported purifier backend: {backend}")

    return Purify(args, config, type, dataset)
