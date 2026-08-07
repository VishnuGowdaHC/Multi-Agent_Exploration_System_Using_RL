Shader "Custom/FogOfWarSmoke"
{
    Properties
    {
        _VisionMask ("Vision Mask", 2D) = "black" {}
        _SmokeColor ("Smoke Color", Color) = (0.02, 0.02, 0.03, 1)
        _GlowColor ("Torch Glow Color", Color) = (1.0, 0.55, 0.2, 1)
        _NoiseScale ("Noise Scale", Float) = 8.0
        _ScrollSpeed1 ("Scroll Speed 1", Vector) = (0.03, 0.02, 0, 0)
        _ScrollSpeed2 ("Scroll Speed 2", Vector) = (-0.02, 0.015, 0, 0)
        _EdgeSoftness ("Edge Softness", Range(0.01, 1)) = 0.35
        _SmokeDensity ("Smoke Density", Range(0,1)) = 0.9
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" "RenderPipeline"="UniversalPipeline" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes { float4 positionOS : POSITION; float2 uv : TEXCOORD0; };
            struct Varyings { float4 positionHCS : SV_POSITION; float2 uv : TEXCOORD0; };

            TEXTURE2D(_VisionMask); SAMPLER(sampler_VisionMask);
            float4 _SmokeColor;
            float4 _GlowColor;
            float _NoiseScale;
            float4 _ScrollSpeed1;
            float4 _ScrollSpeed2;
            float _EdgeSoftness;
            float _SmokeDensity;

            Varyings vert(Attributes v)
            {
                Varyings o;
                o.positionHCS = TransformObjectToHClip(v.positionOS.xyz);
                o.uv = v.uv;
                return o;
            }

            float hash(float2 p)
            {
                return frac(sin(dot(p, float2(127.1,311.7))) * 43758.5453123);
            }

            float noise(float2 p)
            {
                float2 i = floor(p);
                float2 f = frac(p);
                float a = hash(i);
                float b = hash(i + float2(1,0));
                float c = hash(i + float2(0,1));
                float d = hash(i + float2(1,1));
                float2 u = f * f * (3.0 - 2.0 * f);
                return lerp(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
            }

            half4 frag(Varyings i) : SV_Target
            {
                float vision = SAMPLE_TEXTURE2D(_VisionMask, sampler_VisionMask, i.uv).r;

                // Only bother with noise distortion near an actual boundary —
                // fully unexplored (vision≈0) or fully explored (vision≈1) cells
                // should never be affected by noise at all.
                float nearBoundary = 1.0 - abs(vision * 2.0 - 1.0); // peaks at vision=0.5, 0 at vision=0 or 1

                float2 uv1 = i.uv * _NoiseScale + _Time.y * _ScrollSpeed1.xy;
                float2 uv2 = i.uv * _NoiseScale * 1.7 + _Time.y * _ScrollSpeed2.xy;
                float n = noise(uv1) * 0.6 + noise(uv2) * 0.4;

                float turbulentVision = saturate(vision + (n - 0.5) * _EdgeSoftness * nearBoundary);

                float alpha = (1.0 - turbulentVision) * _SmokeDensity;
                alpha *= lerp(0.85, 1.0, n); // subtler internal density variation

                float edgeGlow = (smoothstep(0.15, 0.55, turbulentVision) - smoothstep(0.55, 0.9, turbulentVision)) * nearBoundary;
                float3 col = lerp(_SmokeColor.rgb, _GlowColor.rgb, saturate(edgeGlow * 1.5));

                return half4(col, alpha);
            }
            ENDHLSL
        }
    }
}