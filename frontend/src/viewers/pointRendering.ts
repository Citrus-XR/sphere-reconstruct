import * as THREE from 'three'

export const createCircularPointMaterial = (multisampled: boolean) => {
  const material = new THREE.PointsMaterial({
    vertexColors: true,
    sizeAttenuation: false,
    toneMapped: false,
    depthTest: true,
    depthWrite: true,
    alphaToCoverage: multisampled,
    transparent: !multisampled,
  })
  material.onBeforeCompile = shader => {
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <color_fragment>',
      `#include <color_fragment>
      float circleDistance = length(gl_PointCoord - vec2(0.5));
      float circleEdge = max(fwidth(circleDistance), 0.0001);
      diffuseColor.a *= 1.0 - smoothstep(0.5 - circleEdge, 0.5, circleDistance);
      if (diffuseColor.a <= 0.0) discard;`,
    )
  }
  material.customProgramCacheKey = () => 'circular-points-v1'
  return material
}
