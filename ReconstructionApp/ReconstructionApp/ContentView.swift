//
//  ContentView.swift
//  ReconstructionApp
//
//  Created by Yuxuan Zhang on 29/11/24.
//

import SwiftUI

struct ContentView: View {
    @State private var modelURL: URL?
    
    var body: some View {
        VStack {
            MainView()
            
            Spacer()
            
            BottomView()
        }
    }
}
