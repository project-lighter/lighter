import fire
import yaml 

from monai.bundle.scripts import run


def interface():
    fire.Fire({
        "fit": fit,
        "validate": validate,
        "predict": predict,
        "test": test,
        "tune": tune
    })

def fit(**kwargs):
    fit_config = """
        fit:
            _method_: >
                $@trainer.fit(model=@fit#model,
                            ckpt_path=@fit#ckpt_path)
            model: "@system"
            ckpt_path: null
    """
    run("fit", **yaml.safe_load(fit_config), **kwargs)

def validate(**kwargs):
    validate_config = """
        validate:
            _method_: > 
                $@trainer.validate(model=@validate#model,
                                ckpt_path=@validate#ckpt_path,
                                verbose=@validate#verbose)
            model: "@system"
            ckpt_path: null
            verbose: True
    """
    run("validate", **yaml.safe_load(validate_config), **kwargs)

def predict(**kwargs):
    predict_config = """
        predict:
            _method_: >
                $@trainer.predict(model=@predict#model,
                                ckpt_path=@predict#ckpt_path)
            model: "@system"
            ckpt_path: null
    """
    run("predict", **yaml.safe_load(predict_config), **kwargs)

def test(**kwargs):
    test_config = """
        test:
            _method_: >
                $@trainer.test(model=@test#model,
                            ckpt_path=@test#ckpt_path,
                            verbose=@test#verbose)
            model: "@system"
            ckpt_path: null
            verbose: True
    """
    run("test", **yaml.safe_load(test_config), **kwargs)

def tune(**kwargs):
    tune_config = """
        tune:
            _method_: > 
                $@trainer.tune(model=@tune#model,
                            ckpt_path=@tune#ckpt_path,
                            scale_batch_size_kwargs=@tune#scale_batch_size_kwargs,
                            lr_find_kwargs=@tune#lr_find_kwargs,
                            method=@tune#method)
            model: "@system"
            ckpt_path: null
            scale_batch_size_kwargs: null
            lr_find_kwargs: null
            method: fit
    """
    run("tune", **yaml.safe_load(tune_config), **kwargs)
