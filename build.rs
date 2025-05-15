use std::{env, fs::File, io::Write, path::PathBuf};

use cargo_toml::Manifest;

const OPENCV_FEATURE_PREFIX: &str = "opencv-";
const OPENCV_DEPENDENCY_PREFIX: &str = "opencv_";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let manifest = Manifest::from_path("./Cargo.toml")?;

    let mut out_file = File::create(out_dir.join("opencv.rs"))?;
    for (feature, _) in manifest.features {
        let Some(stripped) = feature.strip_prefix(OPENCV_FEATURE_PREFIX) else {
            continue;
        };

        let dependency = format!("{}{}", OPENCV_DEPENDENCY_PREFIX, stripped);

        writeln!(
            &mut out_file,
            r#"
                #[cfg(feature = "{feature}")]
                pub(crate) use {dependency}::*;
            "#
        )?;
    }

    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
