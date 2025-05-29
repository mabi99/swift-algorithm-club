//  Based on:
//  Octree.swift
//  Swift Algorithm Club
//
//  Written for Swift Algorithm Club by Jaap Wijnen *Heavily inspired by
//  Timur Galimov's Quadtree implementation and Apple's GKOctree implementation
//
//  https://github.com/kodecocodes/swift-algorithm-club/blob/master/Octree/README.md
//
//  Corrected, refined and improved by Marco Binder (Heidelberg, Germany), 2025
//  using ChatGPT as a coding buddy. BUG FIXES: bug removed in tryAdd, where it was
//  implicitly expected that elements is never nil (which it was upon subdivision).
//  REFINEMENTS: improved efficiency in Box .contains(simd3_double), including using
//  an epsilon to prevent unneccessary subdivision upon rounding errors; several
//  other steps refined. IMPROVEMENTS: tree now autocollapses upon removing elements,
//  turning empty leafs into internal nodes again; accepts elements with box-regions
//  instead of point positions, similar to the Apple GKOctree implementation; now
//  respects minimumCellSize from initializer, saves it and subdivides only until
//  reaching minimum size, after which it bulks elements in single leave.

import Foundation
import SIMD


public struct MBBox: CustomStringConvertible {
    public var boxMin: SIMD3<Double>
    public var boxMax: SIMD3<Double>
    
    public init(boxMin: SIMD3<Double>, boxMax: SIMD3<Double>) {
        self.boxMin = boxMin
        self.boxMax = boxMax
    }
    
    public var boxSize: SIMD3<Double> {
        return boxMax - boxMin
    }
    
    var halfBoxSize: SIMD3<Double> {
        return boxSize/2
    }
    
    var frontLeftTop: MBBox {
        let boxMin = self.boxMin + SIMD3<Double>(0, halfBoxSize.y, halfBoxSize.z)
        let boxMax = self.boxMax - SIMD3<Double>(halfBoxSize.x, 0, 0)
        return MBBox(boxMin: boxMin, boxMax: boxMax)
    }
    var frontLeftBottom: MBBox {
        let boxMin = self.boxMin + SIMD3<Double>(0, 0, halfBoxSize.z)
        let boxMax = self.boxMax - SIMD3<Double>(halfBoxSize.x, halfBoxSize.y, 0)
        return MBBox(boxMin: boxMin, boxMax: boxMax)
    }
    var frontRightTop: MBBox {
        let boxMin = self.boxMin + SIMD3<Double>(halfBoxSize.x, halfBoxSize.y, halfBoxSize.z)
        let boxMax = self.boxMax - SIMD3<Double>(0, 0, 0)
        return MBBox(boxMin: boxMin, boxMax: boxMax)
    }
    var frontRightBottom: MBBox {
        let boxMin = self.boxMin + SIMD3<Double>(halfBoxSize.x, 0, halfBoxSize.z)
        let boxMax = self.boxMax - SIMD3<Double>(0, halfBoxSize.y, 0)
        return MBBox(boxMin: boxMin, boxMax: boxMax)
    }
    var backLeftTop: MBBox {
        let boxMin = self.boxMin + SIMD3<Double>(0, halfBoxSize.y, 0)
        let boxMax = self.boxMax - SIMD3<Double>(halfBoxSize.x, 0, halfBoxSize.z)
        return MBBox(boxMin: boxMin, boxMax: boxMax)
    }
    var backLeftBottom: MBBox {
        let boxMin = self.boxMin + SIMD3<Double>(0, 0, 0)
        let boxMax = self.boxMax - SIMD3<Double>(halfBoxSize.x, halfBoxSize.y, halfBoxSize.z)
        return MBBox(boxMin: boxMin, boxMax: boxMax)
    }
    var backRightTop: MBBox {
        let boxMin = self.boxMin + SIMD3<Double>(halfBoxSize.x, halfBoxSize.y, 0)
        let boxMax = self.boxMax - SIMD3<Double>(0, 0, halfBoxSize.z)
        return MBBox(boxMin: boxMin, boxMax: boxMax)
    }
    var backRightBottom: MBBox {
        let boxMin = self.boxMin + SIMD3<Double>(halfBoxSize.x, 0, 0)
        let boxMax = self.boxMax - SIMD3<Double>(0, halfBoxSize.y, halfBoxSize.z)
        return MBBox(boxMin: boxMin, boxMax: boxMax)
    }
    
    /// Robust against floating-point rounding errors
    public func contains(_ point: SIMD3<Double>, epsilon: Double = 1e-10) -> Bool {
        return (boxMin.x - epsilon <= point.x && point.x <= boxMax.x + epsilon) &&
               (boxMin.y - epsilon <= point.y && point.y <= boxMax.y + epsilon) &&
               (boxMin.z - epsilon <= point.z && point.z <= boxMax.z + epsilon)
    }

    /// Compatible legacy overload without epsilon
    public func contains(_ point: SIMD3<Double>) -> Bool {
        contains(point, epsilon: 1e-10)
    }
    
    
    public func isContained(in box: MBBox) -> Bool {
        return
            self.boxMin.x >= box.boxMin.x &&
                self.boxMin.y >= box.boxMin.y &&
                self.boxMin.z >= box.boxMin.z &&
                self.boxMax.x <= box.boxMax.x &&
                self.boxMax.y <= box.boxMax.y &&
                self.boxMax.z <= box.boxMax.z
    }
    
    public func intersects(_ other: MBBox) -> Bool {
        return !(boxMax.x < other.boxMin.x ||
                 boxMin.x > other.boxMax.x ||
                 boxMax.y < other.boxMin.y ||
                 boxMin.y > other.boxMax.y ||
                 boxMax.z < other.boxMin.z ||
                 boxMin.z > other.boxMax.z)
    }
    
    public var description: String {
        return "Box from:\(boxMin) to:\(boxMax)"
    }
}

public class MBOctreeNode<T: Hashable>: CustomStringConvertible {
    let box: MBBox
    let minimumCellSize: Double
    private var point: SIMD3<Double>?
    private var elements: [T]?
    var type: NodeType = .leaf
    /// Stores elements that occupy entire spatial regions
    private var regionElements: [(element: T, region: MBBox)] = []
    /// Helper to access child nodes as an array
    private var childrenNodes: [MBOctreeNode<T>] {
        guard case .internal(let c) = type else { return [] }
        return [
            c.frontLeftTop, c.frontLeftBottom, c.frontRightTop, c.frontRightBottom,
            c.backLeftTop,  c.backLeftBottom,  c.backRightTop,  c.backRightBottom
        ]
    }
    
    enum NodeType {
        case leaf
        case `internal`(children: Children)
    }
    
    public var description: String {
        switch type {
        case .leaf:
            return "leaf node with \(box) elements: \(String(describing: elements))"
        case .internal:
            return "internal node with \(box)"
        }
    }
    
    var recursiveDescription: String {
        return recursiveDescription(withTabCount: 0)
    }
    
    private func recursiveDescription(withTabCount count: Int) -> String {
        let indent = String(repeating: "\t", count: count)
        var result = "\(indent)" + description + "\n"
        switch type {
        case .internal(let children):
            for child in children {
                result += child.recursiveDescription(withTabCount: count + 1)
            }
        default:
            break
        }
        return result
    }
    
    struct Children: Sequence {
        let frontLeftTop: MBOctreeNode
        let frontLeftBottom: MBOctreeNode
        let frontRightTop: MBOctreeNode
        let frontRightBottom: MBOctreeNode
        let backLeftTop: MBOctreeNode
        let backLeftBottom: MBOctreeNode
        let backRightTop: MBOctreeNode
        let backRightBottom: MBOctreeNode
        
        init(parentNode: MBOctreeNode) {
            frontLeftTop = MBOctreeNode(box: parentNode.box.frontLeftTop, minimumCellSize: parentNode.minimumCellSize)
            frontLeftBottom = MBOctreeNode(box: parentNode.box.frontLeftBottom, minimumCellSize: parentNode.minimumCellSize)
            frontRightTop = MBOctreeNode(box: parentNode.box.frontRightTop, minimumCellSize: parentNode.minimumCellSize)
            frontRightBottom = MBOctreeNode(box: parentNode.box.frontRightBottom, minimumCellSize: parentNode.minimumCellSize)
            backLeftTop = MBOctreeNode(box: parentNode.box.backLeftTop, minimumCellSize: parentNode.minimumCellSize)
            backLeftBottom = MBOctreeNode(box: parentNode.box.backLeftBottom, minimumCellSize: parentNode.minimumCellSize)
            backRightTop = MBOctreeNode(box: parentNode.box.backRightTop, minimumCellSize: parentNode.minimumCellSize)
            backRightBottom = MBOctreeNode(box: parentNode.box.backRightBottom, minimumCellSize: parentNode.minimumCellSize)
        }
        
        struct ChildrenIterator: IteratorProtocol {
            var index = 0
            let children: Children
            
            init(children: Children) {
                self.children = children
            }
            
            mutating func next() -> MBOctreeNode? {
                defer { index += 1 }
                switch index {
                case 0: return children.frontLeftTop
                case 1: return children.frontLeftBottom
                case 2: return children.frontRightTop
                case 3: return children.frontRightBottom
                case 4: return children.backLeftTop
                case 5: return children.backLeftBottom
                case 6: return children.backRightTop
                case 7: return children.backRightBottom
                default: return nil
                }
            }
        }
        
        func makeIterator() -> ChildrenIterator {
            return ChildrenIterator(children: self)
        }
    }
    
    init(box: MBBox, minimumCellSize: Double) {
        self.box = box
        self.minimumCellSize = minimumCellSize
    }
    
    @discardableResult
    func add(_ element: T, at point: SIMD3<Double>) -> MBOctreeNode? {
        return tryAdd(element, at: point)
    }

    /// Adds an element that occupies the entire region of space
    @discardableResult
    func add(_ element: T, in region: MBBox) -> MBOctreeNode? {
        return tryAdd(element, in: region)
    }

    private func tryAdd(_ element: T, at point: SIMD3<Double>) -> MBOctreeNode? {
        if !box.contains(point, epsilon: 1e-10) { return nil }
        
        switch type {
        case .internal:
            // pass the point to one of the children
            for child in childrenNodes {
                if let child = child.tryAdd(element, at: point) {
                    return child
                }
            }
            // Fallback: kein Child angenommen – Loggen und überspringen
            NSLog("Warning: point \(point) in \(box) but no child accepted it, skipping element \(element)")
            return nil
        case .leaf:
            let maxSize = max(box.boxSize.x, box.boxSize.y, box.boxSize.z)
            if maxSize / 2.0 < minimumCellSize {
                self.elements?.append(element) ?? { self.elements = [element] }()
                return self
            }
            if self.point != nil {
                // leaf already has an asigned point
                if self.point == point {
                    self.elements?.append(element) ?? { self.elements = [element] }()
                    return self
                } else {
                    return subdivide(adding: element, at: point)
                }
            } else {
                self.elements = [element]
                self.point = point
                return self
            }
        }
    }

    func tryAdd(_ element: T, in region: MBBox) -> MBOctreeNode? {
        // nur einfügen, wenn die Region in diesen Knoten passt
        guard region.isContained(in: box) else { return nil }
        switch type {
        case .internal:
            // wenn ein Kind die Region komplett enthält, dort weiterleiten
            for child in childrenNodes {
                if region.isContained(in: child.box) {
                    return child.tryAdd(element, in: region)
                }
            }
            // ansonsten hier speichern
            regionElements.append((element: element, region: region))
            return self
        case .leaf:
            // Convert leaf to internal node to redistribute point-based elements
            type = .internal(children: Children(parentNode: self))
            if let p = point, let els = elements {
                for e in els {
                    _ = tryAdd(e, at: p)
                }
            }
            // Reset leaf status
            elements = nil
            point = nil
            // dann Region hier einfügen
            return tryAdd(element, in: region)
        }
    }
    
    func add(_ elements: [T], at point: SIMD3<Double>) {
        for element in elements {
            self.add(element, at: point)
        }
    }
    
    @discardableResult
    func remove(_ element: T) -> Bool {
        // Region-based removal: remove from regionElements if present
        if let idx = regionElements.firstIndex(where: { $0.element == element }) {
            regionElements.remove(at: idx)
            return true
        }
        
        switch type {
        case .leaf:
            guard let index = self.elements?.firstIndex(of: element) else { return false }
            self.elements?.remove(at: index)
            if self.elements?.isEmpty == true {
                self.elements = nil
                self.point = nil
            }
            return true
        case .internal:
            for child in childrenNodes {
                if child.remove(element) {
                    collapseIfEmpty()
                    return true
                }
            }
            return false
        }
    }
    
    func elements(at point: SIMD3<Double>) -> [T]? {
        var result: [T] = []
        switch type {
        case .leaf:
            if self.point == point {
                if let els = self.elements {
                    result.append(contentsOf: els)
                }
            }
        case .internal:
            for child in childrenNodes {
                if child.box.contains(point) {
                    if let els = child.elements(at: point) {
                        result.append(contentsOf: els)
                    }
                }
            }
        }
        // Include region-based elements
        for (elem, region) in regionElements where region.contains(point) {
            result.append(elem)
        }
        return result.isEmpty ? nil : result
    }
    
    func elements(in box: MBBox) -> [T]? {
        var values: [T] = []
        switch type {
        case .leaf:
            // check if leaf has an assigned point
            if let point = self.point {
                // check if point is inside given box
                if box.contains(point) {
                    values += elements ?? []
                }
            }
        case .internal:
            // Reserve approximate capacity for efficiency
            values.reserveCapacity(values.count + childrenNodes.count)
            for child in childrenNodes {
                if child.box.isContained(in: box) {
                    // Complete subtree case: collect all elements
                    values += child.collectAllElements()
                } else if child.box.intersects(box) {
                    // Partial overlap case: recursive query
                    values += child.elements(in: box) ?? []
                }
                // child does not contain any part of given box
            }
        }
        // Also include region-based elements
        for (elem, region) in regionElements where region.intersects(box) {
            values.append(elem)
        }
        if values.isEmpty { return nil }
        return values
    }
    
    private func subdivide(adding element: T, at point: SIMD3<Double>) -> MBOctreeNode? {
        guard let oldElements = self.elements, let oldPoint = self.point else {
            NSLog("⚠️ Inconsistent state: subdivide on empty leaf")
            self.elements = [element]
            self.point = point
            return self
        }
        type = .internal(children: Children(parentNode: self))
        self.elements = nil
        self.point = nil
        self.add(oldElements, at: oldPoint)
        return self.add(element, at: point)
    }

    /// Collapses internal node to empty leaf if all children are empty leaves
    private func collapseIfEmpty() {
        guard case .internal(let children) = type else { return }
        var allEmpty = true
        for child in children {
            if case .leaf = child.type {
                if let els = child.elements, !els.isEmpty {
                    allEmpty = false
                    break
                }
            } else {
                allEmpty = false
                break
            }
        }
        if allEmpty {
            type = .leaf
            elements = nil
            point = nil
        }
    }

    /// Sammelt alle Elemente dieses Knotens und seiner Nachkommen
    func collectAllElements() -> [T] {
        var result: [T] = []
        switch type {
        case .leaf:
            if let els = elements {
                result.append(contentsOf: els)
            }
        case .internal(let children):
            for child in children {
                result.append(contentsOf: child.collectAllElements())
            }
        }
        return result
    }
}

public class MBOctree<T: Hashable>: CustomStringConvertible {
    private let minimumCellSize: Double
    var root: MBOctreeNode<T>
    
    public var description: String {
        return "Octree\n" + root.recursiveDescription
    }
    
    public init(boundingBox: MBBox, minimumCellSize: Double) {
        self.minimumCellSize = minimumCellSize
        root = MBOctreeNode<T>(box: boundingBox, minimumCellSize: minimumCellSize)
    }
    
    @discardableResult
    public func add(_ element: T, at point: SIMD3<Double>) -> MBOctreeNode<T>? {
        return root.add(element, at: point)
    }

    /// Adds an element that occupies a region of space
    @discardableResult
    public func add(_ element: T, in region: MBBox) -> MBOctreeNode<T>? {
        return root.add(element, in: region)
    }
    
    @discardableResult
    public func remove(_ element: T, using node: MBOctreeNode<T>) -> Bool {
        return node.remove(element)
    }
    
    @discardableResult
    public func remove(_ element: T) -> Bool {
        return root.remove(element)
    }
    
    public func elements(at point: SIMD3<Double>) -> [T]? {
        return root.elements(at: point)
    }
    
    public func elements(in box: MBBox) -> [T]? {
        precondition(box.isContained(in: root.box), "box is outside of octree bounds")
        return root.elements(in: box)
    }
}
