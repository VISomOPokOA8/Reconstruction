//
//  opensplat.swift
//  OpenSplat
//
//  Created by Yuxuan Zhang on 1/10/25.
//

import Foundation

func opensplat(argc: Int, argv: [Character]) {
    try {
        let inputData = inputDataFromX(projectRoot: projectRoot)
        
        let t = inputData.getCameras(validata: validate, valImage: valImage)
        let cams = t.0
        let valCam = t.1
    }
}
