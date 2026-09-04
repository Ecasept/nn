const std = @import("std");

const funcs = struct {
    fn sigmoid(x: f32) f32 {
        return 1.0 / (1 + std.math.exp(-x));
    }
    fn sigmoidDeriv(x: f32) f32 {
        return sigmoid(x) * (1 - sigmoid(x));
    }
    fn sigmoidDerivFromActivation(x: f32) f32 {
        return x * (1 - x);
    }
    fn relu(x: f32) f32 {
        return if (x > 0) x else 0;
    }
    fn reluDeriv(x: f32) f32 {
        return if (x > 0) 1 else 0;
    }
};

pub fn Activations() type {
    return struct {
        pub fn sigmoid() Activation {
            return Activation{
                .activation = &funcs.sigmoid,
                .activationDeriv = &funcs.sigmoidDeriv,
                .activationDerivFromActivation = &funcs.sigmoidDerivFromActivation,
                .weightInitialization = .xavier,
            };
        }
        pub fn relu() Activation {
            return Activation{
                .activation = &funcs.relu,
                .activationDeriv = &funcs.reluDeriv,
                .activationDerivFromActivation = &funcs.reluDeriv,
                .weightInitialization = .he,
            };
        }
    };
}

pub const Activation = struct {
    activation: *const fn (f32) f32,
    activationDeriv: *const fn (f32) f32,
    activationDerivFromActivation: *const fn (f32) f32,
    weightInitialization: WeightInitialization,
};

pub const WeightInitialization = enum {
    xavier,
    he,
};
