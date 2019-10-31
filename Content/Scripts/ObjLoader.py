# If we separate this to two file, and it seems that ue can't get the changes from another file
# only handle vertex and index buffer
class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.faces = []

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = [float(values[1]), float(values[2]), float(values[3])]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append([v[0], v[1], v[2]])
            elif values[0] == 'vn':
                vn = [float(values[1]), float(values[2]), float(values[3])]
                if swapyz:
                    vn = vn[0], vn[2], vn[1]
                self.normals.append([vn[0], vn[1], vn[2]])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    self.faces.append(int(w[0])-1)
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)

        print("Load obj vertices***")
        print(len(self.vertices))