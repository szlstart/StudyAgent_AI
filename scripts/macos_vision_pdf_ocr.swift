#!/usr/bin/env swift

import AppKit
import Foundation
import PDFKit
import Vision

struct OCRLine: Codable {
    let page: Int
    let text: String
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

guard CommandLine.arguments.count >= 3 else {
    fail("用法: swift scripts/macos_vision_pdf_ocr.swift <input.pdf> <output.jsonl> [max_pages]")
}

let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
let maxPages = CommandLine.arguments.count >= 4 ? Int(CommandLine.arguments[3]) : nil

guard let document = PDFDocument(url: inputURL) else {
    fail("无法打开 PDF: \(inputURL.path)")
}

var completed = Set<Int>()
if let existing = try? String(contentsOf: outputURL, encoding: .utf8) {
    let decoder = JSONDecoder()
    for line in existing.split(separator: "\n") {
        if let data = line.data(using: .utf8),
           let record = try? decoder.decode(OCRLine.self, from: data) {
            completed.insert(record.page)
        }
    }
}

if !FileManager.default.fileExists(atPath: outputURL.path) {
    FileManager.default.createFile(atPath: outputURL.path, contents: nil)
}
guard let output = try? FileHandle(forWritingTo: outputURL) else {
    fail("无法写入: \(outputURL.path)")
}
defer { try? output.close() }
try? output.seekToEnd()

let encoder = JSONEncoder()
encoder.outputFormatting = [.withoutEscapingSlashes]
let pageLimit = min(document.pageCount, maxPages ?? document.pageCount)

for pageIndex in 0..<pageLimit {
    let pageNumber = pageIndex + 1
    if completed.contains(pageNumber) { continue }

    autoreleasepool {
        guard let page = document.page(at: pageIndex) else { return }
        let bounds = page.bounds(for: .cropBox)
        let scale = min(3200.0 / max(bounds.height, 1), 2400.0 / max(bounds.width, 1))
        let size = NSSize(width: max(1200, bounds.width * scale), height: max(1600, bounds.height * scale))
        let image = page.thumbnail(of: size, for: .cropBox)
        var rect = NSRect(origin: .zero, size: image.size)
        guard let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
            FileHandle.standardError.write("第 \(pageNumber) 页渲染失败\n".data(using: .utf8)!)
            return
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["zh-Hans", "en-US"]
        request.usesLanguageCorrection = true
        request.minimumTextHeight = 0.004

        do {
            try VNImageRequestHandler(cgImage: cgImage, options: [:]).perform([request])
            let observations = (request.results ?? []).sorted { lhs, rhs in
                if abs(lhs.boundingBox.midY - rhs.boundingBox.midY) > 0.008 {
                    return lhs.boundingBox.midY > rhs.boundingBox.midY
                }
                return lhs.boundingBox.minX < rhs.boundingBox.minX
            }
            let text = observations.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
            let record = OCRLine(page: pageNumber, text: text)
            if let data = try? encoder.encode(record) {
                output.write(data)
                output.write(Data("\n".utf8))
                try? output.synchronize()
            }
            print("OCR \(pageNumber)/\(pageLimit) chars=\(text.count)")
        } catch {
            FileHandle.standardError.write("第 \(pageNumber) 页 OCR 失败: \(error)\n".data(using: .utf8)!)
        }
    }
}
