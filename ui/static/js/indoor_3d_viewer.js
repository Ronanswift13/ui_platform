/**
 * 室内监测中心 3D数字孪生可视化模块 V1.0
 * =============================================
 *
 * 基于Three.js实现的3D可视化系统
 *
 * 功能:
 * 1. 三维场景渲染 - SLAM点云/OBJ模型加载
 * 2. 实时对象渲染 - 人员、动物、火焰等
 * 3. 热力图叠加 - 温度、气体浓度云图
 * 4. 轨迹动画 - 运动方向和速度展示
 * 5. 交互式告警 - 点击定位、回放、处理流程
 *
 * 输变电激光星芒破夜绘明监测平台
 * 版本: 1.0.0
 */

// =============================================================================
// Three.js 3D场景管理器
// =============================================================================
class Indoor3DViewer {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = null;
        this.options = {
            backgroundColor: 0x0a0a1a,
            ambientLightColor: 0x404040,
            directionalLightColor: 0xffffff,
            gridSize: 20,
            gridDivisions: 40,
            cameraFov: 60,
            cameraNear: 0.1,
            cameraFar: 1000,
            enableShadows: true,
            enablePostProcessing: false,
            ...options
        };

        // Three.js核心对象
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.raycaster = null;
        this.mouse = null;

        // 场景对象组
        this.groups = {
            environment: null,  // 环境模型
            pointCloud: null,   // 点云
            persons: null,      // 人员
            animals: null,      // 动物
            fires: null,        // 火焰
            zones: null,        // 禁区
            trajectories: null, // 轨迹
            heatmap: null,      // 热力图
            labels: null,       // 标签
        };

        // 数据状态
        this.state = {
            persons: new Map(),
            animals: new Map(),
            fires: new Map(),
            zones: new Map(),
            trajectories: new Map(),
            alarms: [],
            selectedObject: null,
            heatmapData: null,
            pointCloudData: null,
        };

        // 动画
        this.animationId = null;
        this.clock = null;
        this.mixers = [];

        // 事件回调
        this.callbacks = {
            onObjectClick: null,
            onAlarmClick: null,
            onZoneEnter: null,
            onZoneExit: null,
        };

        // 初始化
        this._init();
    }

    // =========================================================================
    // 初始化
    // =========================================================================
    _init() {
        this.container = document.getElementById(this.containerId);
        if (!this.container) {
            console.error(`[Indoor3D] 容器 ${this.containerId} 不存在`);
            return;
        }

        // 检查Three.js是否加载
        if (typeof THREE === 'undefined') {
            console.error('[Indoor3D] Three.js 未加载');
            return;
        }

        this._initScene();
        this._initCamera();
        this._initRenderer();
        this._initLights();
        this._initControls();
        this._initGroups();
        this._initRaycaster();
        this._initEnvironment();
        this._bindEvents();

        this.clock = new THREE.Clock();
        this._animate();

        console.log('[Indoor3D] 初始化完成');
    }

    _initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.options.backgroundColor);
        this.scene.fog = new THREE.Fog(this.options.backgroundColor, 50, 200);
    }

    _initCamera() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera = new THREE.PerspectiveCamera(
            this.options.cameraFov,
            width / height,
            this.options.cameraNear,
            this.options.cameraFar
        );
        this.camera.position.set(15, 15, 15);
        this.camera.lookAt(0, 0, 0);
    }

    _initRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        if (this.options.enableShadows) {
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        }

        this.container.appendChild(this.renderer.domElement);
    }

    _initLights() {
        // 环境光
        const ambientLight = new THREE.AmbientLight(
            this.options.ambientLightColor,
            0.6
        );
        this.scene.add(ambientLight);

        // 主方向光
        const directionalLight = new THREE.DirectionalLight(
            this.options.directionalLightColor,
            0.8
        );
        directionalLight.position.set(10, 20, 10);
        directionalLight.castShadow = this.options.enableShadows;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 50;
        this.scene.add(directionalLight);

        // 补光
        const fillLight = new THREE.DirectionalLight(0x4488ff, 0.3);
        fillLight.position.set(-10, 10, -10);
        this.scene.add(fillLight);
    }

    _initControls() {
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.screenSpacePanning = false;
            this.controls.minDistance = 5;
            this.controls.maxDistance = 100;
            this.controls.maxPolarAngle = Math.PI / 2;
        }
    }

    _initGroups() {
        Object.keys(this.groups).forEach(key => {
            this.groups[key] = new THREE.Group();
            this.groups[key].name = key;
            this.scene.add(this.groups[key]);
        });
    }

    _initRaycaster() {
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
    }

    _initEnvironment() {
        // 地面网格
        const gridHelper = new THREE.GridHelper(
            this.options.gridSize,
            this.options.gridDivisions,
            0x444466,
            0x222244
        );
        this.groups.environment.add(gridHelper);

        // 地面平面
        const groundGeometry = new THREE.PlaneGeometry(
            this.options.gridSize,
            this.options.gridSize
        );
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x111122,
            roughness: 0.8,
            metalness: 0.2,
            transparent: true,
            opacity: 0.8,
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -0.01;
        ground.receiveShadow = true;
        this.groups.environment.add(ground);

        // 坐标轴辅助
        const axesHelper = new THREE.AxesHelper(5);
        this.groups.environment.add(axesHelper);
    }

    _bindEvents() {
        // 窗口大小变化
        window.addEventListener('resize', () => this._onResize());

        // 鼠标点击
        this.renderer.domElement.addEventListener('click', (e) => this._onClick(e));

        // 鼠标移动
        this.renderer.domElement.addEventListener('mousemove', (e) => this._onMouseMove(e));
    }

    // =========================================================================
    // 事件处理
    // =========================================================================
    _onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    _onClick(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);

        // 检测可点击对象
        const clickableGroups = [
            this.groups.persons,
            this.groups.animals,
            this.groups.fires,
            this.groups.zones,
        ];

        const objects = [];
        clickableGroups.forEach(group => {
            group.traverse(child => {
                if (child.isMesh && child.userData.clickable) {
                    objects.push(child);
                }
            });
        });

        const intersects = this.raycaster.intersectObjects(objects, true);

        if (intersects.length > 0) {
            const object = intersects[0].object;
            this._selectObject(object);

            if (this.callbacks.onObjectClick) {
                this.callbacks.onObjectClick(object.userData);
            }
        } else {
            this._deselectObject();
        }
    }

    _onMouseMove(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    }

    _selectObject(object) {
        this._deselectObject();

        this.state.selectedObject = object;

        // 高亮效果
        if (object.material) {
            object.userData.originalEmissive = object.material.emissive?.clone();
            if (object.material.emissive) {
                object.material.emissive.setHex(0x444444);
            }
        }
    }

    _deselectObject() {
        if (this.state.selectedObject) {
            const object = this.state.selectedObject;
            if (object.material && object.userData.originalEmissive) {
                object.material.emissive.copy(object.userData.originalEmissive);
            }
            this.state.selectedObject = null;
        }
    }

    // =========================================================================
    // 动画循环
    // =========================================================================
    _animate() {
        this.animationId = requestAnimationFrame(() => this._animate());

        const delta = this.clock.getDelta();

        // 更新控制器
        if (this.controls) {
            this.controls.update();
        }

        // 更新动画混合器
        this.mixers.forEach(mixer => mixer.update(delta));

        // 更新轨迹动画
        this._updateTrajectoryAnimations(delta);

        // 更新热力图动画
        this._updateHeatmapAnimation(delta);

        // 更新标签朝向
        this._updateLabels();

        // 渲染
        this.renderer.render(this.scene, this.camera);
    }

    _updateTrajectoryAnimations(delta) {
        this.groups.trajectories.children.forEach(trajectory => {
            if (trajectory.userData.animationProgress !== undefined) {
                trajectory.userData.animationProgress += delta * 0.5;
                if (trajectory.userData.animationProgress > 1) {
                    trajectory.userData.animationProgress = 0;
                }

                // 更新箭头位置
                if (trajectory.userData.arrow) {
                    const points = trajectory.userData.points;
                    const progress = trajectory.userData.animationProgress;
                    const index = Math.floor(progress * (points.length - 1));
                    const nextIndex = Math.min(index + 1, points.length - 1);
                    const t = (progress * (points.length - 1)) - index;

                    const pos = new THREE.Vector3().lerpVectors(
                        points[index],
                        points[nextIndex],
                        t
                    );
                    trajectory.userData.arrow.position.copy(pos);
                }
            }
        });
    }

    _updateHeatmapAnimation(delta) {
        if (this.groups.heatmap.children.length > 0) {
            this.groups.heatmap.children.forEach(mesh => {
                if (mesh.material && mesh.material.uniforms) {
                    mesh.material.uniforms.time.value += delta;
                }
            });
        }
    }

    _updateLabels() {
        this.groups.labels.children.forEach(label => {
            if (label.userData.lookAtCamera) {
                label.lookAt(this.camera.position);
            }
        });
    }

    // =========================================================================
    // 点云加载
    // =========================================================================
    loadPointCloud(data, options = {}) {
        const {
            pointSize = 0.05,
            colorMode = 'height', // 'height', 'semantic', 'intensity'
            semanticColors = null,
        } = options;

        // 清除旧点云
        this.clearGroup('pointCloud');

        if (!data || !data.points || data.points.length === 0) {
            console.warn('[Indoor3D] 点云数据为空');
            return;
        }

        const points = data.points;
        const geometry = new THREE.BufferGeometry();

        const positions = new Float32Array(points.length * 3);
        const colors = new Float32Array(points.length * 3);

        // 计算边界
        let minY = Infinity, maxY = -Infinity;
        points.forEach(p => {
            minY = Math.min(minY, p.y || p[1] || 0);
            maxY = Math.max(maxY, p.y || p[1] || 0);
        });
        const heightRange = maxY - minY || 1;

        points.forEach((point, i) => {
            const x = point.x !== undefined ? point.x : point[0] || 0;
            const y = point.y !== undefined ? point.y : point[1] || 0;
            const z = point.z !== undefined ? point.z : point[2] || 0;

            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;

            // 颜色计算
            let color;
            if (colorMode === 'semantic' && point.semantic !== undefined && semanticColors) {
                color = semanticColors[point.semantic] || new THREE.Color(0x888888);
            } else if (colorMode === 'intensity' && point.intensity !== undefined) {
                const intensity = point.intensity;
                color = new THREE.Color().setHSL(0.6, 0.8, intensity);
            } else {
                // 高度着色
                const normalizedHeight = (y - minY) / heightRange;
                color = new THREE.Color().setHSL(0.6 - normalizedHeight * 0.4, 0.8, 0.5);
            }

            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        });

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: pointSize,
            vertexColors: true,
            sizeAttenuation: true,
        });

        const pointCloud = new THREE.Points(geometry, material);
        pointCloud.name = 'pointCloud';
        this.groups.pointCloud.add(pointCloud);

        this.state.pointCloudData = data;
        console.log(`[Indoor3D] 点云加载完成: ${points.length} 点`);
    }

    // =========================================================================
    // OBJ/GLTF/PCD模型加载
    // =========================================================================
    loadModel(url, options = {}) {
        const {
            scale = 1,
            position = { x: 0, y: 0, z: 0 },
            rotation = { x: 0, y: 0, z: 0 },
            pointSize = 0.05,
            pointColor = 0xffffff,
            pointOpacity = 1.0,
        } = options;

        const extension = url.split('?')[0].split('.').pop().toLowerCase();

        if (extension === 'pcd') {
            this._loadPCD(url, scale, position, rotation, {
                pointSize,
                pointColor,
                pointOpacity,
            });
        } else if (extension === 'obj') {
            this._loadOBJ(url, scale, position, rotation);
        } else if (extension === 'gltf' || extension === 'glb') {
            this._loadGLTF(url, scale, position, rotation);
        } else {
            console.error(`[Indoor3D] 不支持的模型格式: ${extension}`);
        }
    }

    _loadOBJ(url, scale, position, rotation) {
        if (typeof THREE.OBJLoader === 'undefined') {
            console.error('[Indoor3D] OBJLoader 未加载');
            return;
        }

        const loader = new THREE.OBJLoader();
        loader.load(
            url,
            (object) => {
                object.scale.setScalar(scale);
                object.position.set(position.x, position.y, position.z);
                object.rotation.set(rotation.x, rotation.y, rotation.z);

                object.traverse(child => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                    }
                });

                this.groups.environment.add(object);
                console.log('[Indoor3D] OBJ模型加载完成');
            },
            (progress) => {
                console.log(`[Indoor3D] OBJ加载进度: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
            },
            (error) => {
                console.error('[Indoor3D] OBJ加载失败:', error);
            }
        );
    }

    _loadGLTF(url, scale, position, rotation) {
        if (typeof THREE.GLTFLoader === 'undefined') {
            console.error('[Indoor3D] GLTFLoader 未加载');
            return;
        }

        const loader = new THREE.GLTFLoader();
        loader.load(
            url,
            (gltf) => {
                const model = gltf.scene;
                model.scale.setScalar(scale);
                model.position.set(position.x, position.y, position.z);
                model.rotation.set(rotation.x, rotation.y, rotation.z);

                model.traverse(child => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                    }
                });

                // 处理动画
                if (gltf.animations && gltf.animations.length > 0) {
                    const mixer = new THREE.AnimationMixer(model);
                    gltf.animations.forEach(clip => {
                        mixer.clipAction(clip).play();
                    });
                    this.mixers.push(mixer);
                }

                this.groups.environment.add(model);
                console.log('[Indoor3D] GLTF模型加载完成');
            },
            (progress) => {
                console.log(`[Indoor3D] GLTF加载进度: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
            },
            (error) => {
                console.error('[Indoor3D] GLTF加载失败:', error);
            }
        );
    }

    _loadPCD(url, scale, position, rotation, options = {}) {
        if (typeof THREE.PCDLoader === 'undefined') {
            console.error('[Indoor3D] PCDLoader 未加载');
            return;
        }

        const {
            pointSize = 0.05,
            pointColor = 0xffffff,
            pointOpacity = 1.0,
        } = options;

        const loader = new THREE.PCDLoader();
        loader.load(
            url,
            (points) => {
                this.clearGroup('pointCloud');

                points.name = 'pointCloud';
                points.scale.setScalar(scale);
                points.position.set(position.x, position.y, position.z);
                points.rotation.set(rotation.x, rotation.y, rotation.z);

                if (points.material) {
                    points.material.size = pointSize;
                    points.material.color = new THREE.Color(pointColor);
                    points.material.transparent = pointOpacity < 1;
                    points.material.opacity = pointOpacity;
                }

                this.groups.pointCloud.add(points);
                this.state.pointCloudData = { source: url };

                console.log('[Indoor3D] PCD点云加载完成');
            },
            (progress) => {
                if (progress.total) {
                    console.log(`[Indoor3D] PCD加载进度: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
                }
            },
            (error) => {
                console.error('[Indoor3D] PCD加载失败:', error);
            }
        );
    }

    // =========================================================================
    // 人员渲染
    // =========================================================================
    updatePersons(persons) {
        if (!Array.isArray(persons)) return;

        const currentIds = new Set(persons.map(p => p.id));

        // 移除不存在的人员
        this.state.persons.forEach((mesh, id) => {
            if (!currentIds.has(id)) {
                this.groups.persons.remove(mesh);
                this.state.persons.delete(id);
            }
        });

        // 更新或添加人员
        persons.forEach(person => {
            const id = person.id;
            let mesh = this.state.persons.get(id);

            // 计算3D位置 (从2D归一化坐标转换)
            const x = (person.x - 0.5) * this.options.gridSize;
            const z = (person.y - 0.5) * this.options.gridSize;
            const y = 0.8; // 人员高度

            if (!mesh) {
                // 创建新人员
                mesh = this._createPersonMesh(person);
                this.groups.persons.add(mesh);
                this.state.persons.set(id, mesh);
            }

            // 更新位置 (平滑过渡)
            this._animatePosition(mesh, { x, y, z });

            // 更新状态颜色
            this._updatePersonStatus(mesh, person);

            // 更新用户数据
            mesh.userData = {
                ...mesh.userData,
                ...person,
                clickable: true,
                type: 'person',
            };
        });
    }

    _createPersonMesh(person) {
        const group = new THREE.Group();

        // 身体 (胶囊形状)
        const bodyGeometry = new THREE.CapsuleGeometry(0.3, 0.8, 4, 8);
        const bodyMaterial = new THREE.MeshStandardMaterial({
            color: this._getStatusColor(person.status),
            roughness: 0.5,
            metalness: 0.1,
        });
        const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
        body.position.y = 0.7;
        body.castShadow = true;
        group.add(body);

        // 头部
        const headGeometry = new THREE.SphereGeometry(0.25, 16, 16);
        const headMaterial = new THREE.MeshStandardMaterial({
            color: 0xffcc99,
            roughness: 0.6,
        });
        const head = new THREE.Mesh(headGeometry, headMaterial);
        head.position.y = 1.5;
        head.castShadow = true;
        group.add(head);

        // 状态光环
        const ringGeometry = new THREE.RingGeometry(0.4, 0.5, 32);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: this._getStatusColor(person.status),
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.6,
        });
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.01;
        group.add(ring);

        // 名称标签
        const label = this._createTextLabel(person.name || `Person-${person.id}`);
        label.position.y = 2;
        group.add(label);

        group.userData.body = body;
        group.userData.ring = ring;
        group.userData.label = label;

        return group;
    }

    _updatePersonStatus(mesh, person) {
        const color = this._getStatusColor(person.status);

        if (mesh.userData.body) {
            mesh.userData.body.material.color.setHex(color);
        }
        if (mesh.userData.ring) {
            mesh.userData.ring.material.color.setHex(color);

            // 危险状态闪烁
            if (person.status === 'danger') {
                const time = Date.now() * 0.005;
                mesh.userData.ring.material.opacity = 0.3 + Math.sin(time) * 0.3;
            } else {
                mesh.userData.ring.material.opacity = 0.6;
            }
        }
    }

    _getStatusColor(status) {
        const colors = {
            'safe': 0x22c55e,
            'warning': 0xeab308,
            'danger': 0xef4444,
            'unknown': 0x6b7280,
        };
        return colors[status] || colors.unknown;
    }

    // =========================================================================
    // 动物渲染
    // =========================================================================
    updateAnimals(animals) {
        if (!Array.isArray(animals)) return;

        const currentIds = new Set(animals.map(a => a.id));

        // 移除不存在的动物
        this.state.animals.forEach((mesh, id) => {
            if (!currentIds.has(id)) {
                this.groups.animals.remove(mesh);
                this.state.animals.delete(id);
            }
        });

        // 更新或添加动物
        animals.forEach(animal => {
            const id = animal.id;
            let mesh = this.state.animals.get(id);

            const x = (animal.x - 0.5) * this.options.gridSize;
            const z = (animal.y - 0.5) * this.options.gridSize;
            const y = 0.2;

            if (!mesh) {
                mesh = this._createAnimalMesh(animal);
                this.groups.animals.add(mesh);
                this.state.animals.set(id, mesh);
            }

            this._animatePosition(mesh, { x, y, z });

            mesh.userData = {
                ...mesh.userData,
                ...animal,
                clickable: true,
                type: 'animal',
            };
        });
    }

    _createAnimalMesh(animal) {
        const group = new THREE.Group();

        // 根据动物类型创建不同形状
        let geometry, color;
        switch (animal.type) {
            case 'mouse':
            case 'rat':
                geometry = new THREE.SphereGeometry(0.15, 8, 8);
                color = 0x8b7355;
                break;
            case 'bird':
                geometry = new THREE.ConeGeometry(0.15, 0.3, 8);
                color = 0x4a90d9;
                break;
            case 'cat':
                geometry = new THREE.BoxGeometry(0.3, 0.25, 0.4);
                color = 0xffa500;
                break;
            case 'snake':
                geometry = new THREE.CylinderGeometry(0.05, 0.05, 0.5, 8);
                color = 0x228b22;
                break;
            default:
                geometry = new THREE.SphereGeometry(0.2, 8, 8);
                color = 0xff6b6b;
        }

        const material = new THREE.MeshStandardMaterial({
            color: color,
            roughness: 0.7,
        });
        const body = new THREE.Mesh(geometry, material);
        body.castShadow = true;
        group.add(body);

        // 警告标记
        const warningGeometry = new THREE.ConeGeometry(0.1, 0.2, 4);
        const warningMaterial = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.8,
        });
        const warning = new THREE.Mesh(warningGeometry, warningMaterial);
        warning.position.y = 0.5;
        warning.rotation.x = Math.PI;
        group.add(warning);

        // 标签
        const label = this._createTextLabel(this._getAnimalTypeName(animal.type));
        label.position.y = 0.7;
        group.add(label);

        return group;
    }

    _getAnimalTypeName(type) {
        const names = {
            'mouse': '老鼠',
            'rat': '老鼠',
            'bird': '鸟类',
            'cat': '猫',
            'snake': '蛇',
            'insect': '昆虫',
            'dog': '狗',
        };
        return names[type] || type;
    }

    // =========================================================================
    // 火焰渲染
    // =========================================================================
    updateFires(fires) {
        if (!Array.isArray(fires)) return;

        const currentIds = new Set(fires.map(f => f.id));

        this.state.fires.forEach((mesh, id) => {
            if (!currentIds.has(id)) {
                this.groups.fires.remove(mesh);
                this.state.fires.delete(id);
            }
        });

        fires.forEach(fire => {
            const id = fire.id;
            let mesh = this.state.fires.get(id);

            const x = (fire.x - 0.5) * this.options.gridSize;
            const z = (fire.y - 0.5) * this.options.gridSize;
            const y = 0;

            if (!mesh) {
                mesh = this._createFireMesh(fire);
                this.groups.fires.add(mesh);
                this.state.fires.set(id, mesh);
            }

            mesh.position.set(x, y, z);

            mesh.userData = {
                ...mesh.userData,
                ...fire,
                clickable: true,
                type: 'fire',
            };
        });
    }

    _createFireMesh(fire) {
        const group = new THREE.Group();

        // 火焰粒子效果 (简化版)
        const fireGeometry = new THREE.ConeGeometry(0.3, 0.8, 8);
        const fireMaterial = new THREE.MeshBasicMaterial({
            color: 0xff4500,
            transparent: true,
            opacity: 0.8,
        });

        for (let i = 0; i < 3; i++) {
            const flame = new THREE.Mesh(fireGeometry, fireMaterial.clone());
            flame.position.y = 0.4 + i * 0.1;
            flame.position.x = (Math.random() - 0.5) * 0.2;
            flame.position.z = (Math.random() - 0.5) * 0.2;
            flame.scale.setScalar(1 - i * 0.2);
            group.add(flame);
        }

        // 光源
        const light = new THREE.PointLight(0xff4500, 1, 5);
        light.position.y = 0.5;
        group.add(light);

        // 烟雾指示
        const smokeGeometry = new THREE.SphereGeometry(0.5, 8, 8);
        const smokeMaterial = new THREE.MeshBasicMaterial({
            color: 0x666666,
            transparent: true,
            opacity: 0.3,
        });
        const smoke = new THREE.Mesh(smokeGeometry, smokeMaterial);
        smoke.position.y = 1.5;
        group.add(smoke);

        return group;
    }

    // =========================================================================
    // 禁区渲染
    // =========================================================================
    updateZones(zones) {
        if (!Array.isArray(zones)) return;

        this.clearGroup('zones');

        zones.forEach(zone => {
            const mesh = this._createZoneMesh(zone);
            this.groups.zones.add(mesh);
            this.state.zones.set(zone.id, mesh);
        });
    }

    _createZoneMesh(zone) {
        const group = new THREE.Group();

        // 将多边形点转换为3D坐标
        const points = zone.polygon.map(p => {
            return new THREE.Vector2(
                (p[0] - 0.5) * this.options.gridSize,
                (p[1] - 0.5) * this.options.gridSize
            );
        });

        const shape = new THREE.Shape(points);

        // 区域地面
        const geometry = new THREE.ShapeGeometry(shape);
        const color = zone.type === 'danger' ? 0xff0000 :
                      zone.type === 'warning' ? 0xffaa00 :
                      zone.type === 'safe' ? 0x00ff00 : 0x0088ff;

        const material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide,
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.y = 0.02;
        group.add(mesh);

        // 边界线
        const lineGeometry = new THREE.BufferGeometry().setFromPoints(
            [...points, points[0]].map(p => new THREE.Vector3(p.x, 0.05, p.y))
        );
        const lineMaterial = new THREE.LineBasicMaterial({
            color: color,
            linewidth: 2,
        });
        const line = new THREE.Line(lineGeometry, lineMaterial);
        group.add(line);

        // 区域标签
        const center = this._getPolygonCenter(points);
        const label = this._createTextLabel(zone.name || zone.id);
        label.position.set(center.x, 0.5, center.y);
        group.add(label);

        group.userData = {
            ...zone,
            clickable: true,
            type: 'zone',
        };

        return group;
    }

    _getPolygonCenter(points) {
        let x = 0, y = 0;
        points.forEach(p => {
            x += p.x;
            y += p.y;
        });
        return { x: x / points.length, y: y / points.length };
    }

    // =========================================================================
    // 轨迹渲染
    // =========================================================================
    updateTrajectories(trajectories) {
        if (!Array.isArray(trajectories)) return;

        this.clearGroup('trajectories');

        trajectories.forEach(trajectory => {
            const mesh = this._createTrajectoryMesh(trajectory);
            this.groups.trajectories.add(mesh);
            this.state.trajectories.set(trajectory.id, mesh);
        });
    }

    _createTrajectoryMesh(trajectory) {
        const group = new THREE.Group();

        // 转换轨迹点
        const points = trajectory.points.map(p => {
            return new THREE.Vector3(
                (p.x - 0.5) * this.options.gridSize,
                0.1,
                (p.y - 0.5) * this.options.gridSize
            );
        });

        if (points.length < 2) return group;

        // 轨迹线
        const curve = new THREE.CatmullRomCurve3(points);
        const curvePoints = curve.getPoints(50);

        const lineGeometry = new THREE.BufferGeometry().setFromPoints(curvePoints);

        // 根据类型设置颜色
        const color = trajectory.type === 'person' ? 0x3b82f6 :
                      trajectory.type === 'animal' ? 0xef4444 :
                      trajectory.type === 'fire' ? 0xff4500 : 0x22c55e;

        const lineMaterial = new THREE.LineBasicMaterial({
            color: color,
            linewidth: 2,
        });

        const line = new THREE.Line(lineGeometry, lineMaterial);
        group.add(line);

        // 方向箭头
        const arrowGeometry = new THREE.ConeGeometry(0.1, 0.3, 8);
        const arrowMaterial = new THREE.MeshBasicMaterial({ color: color });
        const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
        arrow.rotation.x = Math.PI / 2;
        group.add(arrow);

        // 速度指示 (线条粗细或颜色渐变)
        if (trajectory.speed !== undefined) {
            const speedLabel = this._createTextLabel(`${trajectory.speed.toFixed(1)} m/s`);
            speedLabel.position.copy(points[points.length - 1]);
            speedLabel.position.y = 0.5;
            group.add(speedLabel);
        }

        group.userData = {
            points: curvePoints,
            arrow: arrow,
            animationProgress: 0,
            ...trajectory,
        };

        return group;
    }

    // =========================================================================
    // 热力图渲染
    // =========================================================================
    updateHeatmap(heatmapData, options = {}) {
        const {
            type = 'temperature', // 'temperature', 'gas', 'density'
            height = 0.1,
            opacity = 0.6,
            heightOffset = 0,
            clear = true,
            layerId = type,
        } = options;

        if (clear) {
            this.clearGroup('heatmap');
        } else if (layerId) {
            this._clearHeatmapLayer(layerId);
        }

        if (!heatmapData || !heatmapData.data || heatmapData.data.length === 0) return;

        const data = heatmapData.data;
        const rows = data.length;
        const cols = data[0]?.length || 0;
        if (!cols) return;

        const cellWidth = this.options.gridSize / cols;
        const cellHeight = this.options.gridSize / rows;

        // 计算数据范围
        let min = Infinity, max = -Infinity;
        data.forEach(row => {
            row.forEach(val => {
                min = Math.min(min, val);
                max = Math.max(max, val);
            });
        });
        const range = max - min || 1;

        // 创建热力图网格
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const value = data[i][j];
                const normalizedValue = (value - min) / range;

                const geometry = new THREE.BoxGeometry(
                    cellWidth * 0.95,
                    height + normalizedValue * 0.5,
                    cellHeight * 0.95
                );

                const color = this._getHeatmapColor(normalizedValue, type);
                const material = new THREE.MeshBasicMaterial({
                    color: color,
                    transparent: true,
                    opacity: opacity * (0.5 + normalizedValue * 0.5),
                });

                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(
                    (j - cols / 2 + 0.5) * cellWidth,
                    heightOffset + (height + normalizedValue * 0.5) / 2,
                    (i - rows / 2 + 0.5) * cellHeight
                );
                mesh.userData.heatmapLayer = layerId;

                this.groups.heatmap.add(mesh);
            }
        }

        this.state.heatmapData = heatmapData;
    }

    _clearHeatmapLayer(layerId) {
        const group = this.groups.heatmap;
        if (!group) return;

        const toRemove = group.children.filter(child => child.userData?.heatmapLayer === layerId);
        toRemove.forEach(child => {
            group.remove(child);
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
    }

    _getHeatmapColor(value, type) {
        let hue;
        switch (type) {
            case 'temperature':
                // 蓝(冷) -> 红(热)
                hue = (1 - value) * 0.6;
                break;
            case 'gas':
                // 绿(安全) -> 红(危险)
                hue = (1 - value) * 0.33;
                break;
            case 'density':
                // 蓝(稀疏) -> 黄(密集)
                hue = 0.6 - value * 0.4;
                break;
            default:
                hue = (1 - value) * 0.6;
        }

        return new THREE.Color().setHSL(hue, 0.8, 0.5);
    }

    // =========================================================================
    // 文本标签
    // =========================================================================
    _createTextLabel(text, options = {}) {
        const {
            fontSize = 48,
            fontColor = '#ffffff',
            backgroundColor = 'rgba(0, 0, 0, 0.6)',
            padding = 10,
        } = options;

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        context.font = `${fontSize}px Arial`;
        const textWidth = context.measureText(text).width;

        canvas.width = textWidth + padding * 2;
        canvas.height = fontSize + padding * 2;

        // 背景
        context.fillStyle = backgroundColor;
        context.roundRect(0, 0, canvas.width, canvas.height, 8);
        context.fill();

        // 文字
        context.font = `${fontSize}px Arial`;
        context.fillStyle = fontColor;
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(text, canvas.width / 2, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
        });

        const sprite = new THREE.Sprite(material);
        sprite.scale.set(canvas.width / 100, canvas.height / 100, 1);
        sprite.userData.lookAtCamera = true;

        return sprite;
    }

    // =========================================================================
    // 告警定位
    // =========================================================================
    focusOnAlarm(alarm) {
        if (!alarm) return;

        const x = (alarm.x - 0.5) * this.options.gridSize;
        const z = (alarm.y - 0.5) * this.options.gridSize;
        const y = 1;

        // 动画移动相机
        this._animateCameraTo({ x, y: y + 5, z: z + 5 }, { x, y, z });

        // 创建告警标记
        this._createAlarmMarker(alarm, { x, y, z });

        if (this.callbacks.onAlarmClick) {
            this.callbacks.onAlarmClick(alarm);
        }
    }

    _createAlarmMarker(alarm, position) {
        // 移除旧标记
        const oldMarker = this.scene.getObjectByName('alarmMarker');
        if (oldMarker) {
            this.scene.remove(oldMarker);
        }

        const group = new THREE.Group();
        group.name = 'alarmMarker';

        // 脉冲圆环
        const ringGeometry = new THREE.RingGeometry(0.5, 0.7, 32);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.8,
        });
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.1;
        group.add(ring);

        // 垂直光柱
        const beamGeometry = new THREE.CylinderGeometry(0.1, 0.3, 5, 8);
        const beamMaterial = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.4,
        });
        const beam = new THREE.Mesh(beamGeometry, beamMaterial);
        beam.position.y = 2.5;
        group.add(beam);

        // 告警信息标签
        const label = this._createTextLabel(alarm.message || '告警', {
            backgroundColor: 'rgba(255, 0, 0, 0.8)',
        });
        label.position.y = 5.5;
        group.add(label);

        group.position.set(position.x, position.y, position.z);
        this.scene.add(group);

        // 脉冲动画
        const animate = () => {
            const time = Date.now() * 0.003;
            ring.scale.setScalar(1 + Math.sin(time) * 0.3);
            ring.material.opacity = 0.5 + Math.sin(time) * 0.3;
            beam.material.opacity = 0.2 + Math.sin(time) * 0.2;

            if (this.scene.getObjectByName('alarmMarker')) {
                requestAnimationFrame(animate);
            }
        };
        animate();
    }

    _animateCameraTo(targetPosition, lookAtPosition) {
        const startPosition = this.camera.position.clone();
        const startTime = Date.now();
        const duration = 1000;

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = this._easeOutCubic(progress);

            this.camera.position.lerpVectors(
                startPosition,
                new THREE.Vector3(targetPosition.x, targetPosition.y, targetPosition.z),
                eased
            );

            if (this.controls) {
                this.controls.target.set(lookAtPosition.x, lookAtPosition.y, lookAtPosition.z);
            }

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        animate();
    }

    _easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    // =========================================================================
    // 工具方法
    // =========================================================================
    _animatePosition(object, targetPosition, duration = 300) {
        const startPosition = object.position.clone();
        const startTime = Date.now();

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            object.position.lerpVectors(
                startPosition,
                new THREE.Vector3(targetPosition.x, targetPosition.y, targetPosition.z),
                progress
            );

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        animate();
    }

    clearGroup(groupName) {
        const group = this.groups[groupName];
        if (group) {
            while (group.children.length > 0) {
                const child = group.children[0];
                group.remove(child);
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(m => m.dispose());
                    } else {
                        child.material.dispose();
                    }
                }
            }
        }

        // 清除状态
        if (groupName === 'persons') this.state.persons.clear();
        if (groupName === 'animals') this.state.animals.clear();
        if (groupName === 'fires') this.state.fires.clear();
        if (groupName === 'zones') this.state.zones.clear();
        if (groupName === 'trajectories') this.state.trajectories.clear();
    }

    clearAll() {
        Object.keys(this.groups).forEach(key => {
            if (key !== 'environment') {
                this.clearGroup(key);
            }
        });
    }

    // =========================================================================
    // 相机控制
    // =========================================================================
    setCameraPosition(x, y, z) {
        this.camera.position.set(x, y, z);
    }

    setCameraTarget(x, y, z) {
        if (this.controls) {
            this.controls.target.set(x, y, z);
        }
    }

    resetCamera() {
        this.camera.position.set(15, 15, 15);
        if (this.controls) {
            this.controls.target.set(0, 0, 0);
        }
    }

    // =========================================================================
    // 事件回调设置
    // =========================================================================
    onObjectClick(callback) {
        this.callbacks.onObjectClick = callback;
    }

    onAlarmClick(callback) {
        this.callbacks.onAlarmClick = callback;
    }

    // =========================================================================
    // 销毁
    // =========================================================================
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        this.clearAll();

        if (this.renderer) {
            this.renderer.dispose();
            this.container.removeChild(this.renderer.domElement);
        }

        if (this.controls) {
            this.controls.dispose();
        }

        console.log('[Indoor3D] 已销毁');
    }
}

// =============================================================================
// 告警处理管理器
// =============================================================================
class AlarmManager {
    constructor(viewer3d) {
        this.viewer = viewer3d;
        this.alarms = [];
        this.handlers = {
            confirm: null,
            disperse: null,
            report: null,
        };
    }

    addAlarm(alarm) {
        alarm.id = alarm.id || Date.now();
        alarm.timestamp = alarm.timestamp || Date.now();
        alarm.status = 'pending';
        this.alarms.push(alarm);
        return alarm;
    }

    getAlarms(filter = {}) {
        let result = [...this.alarms];

        if (filter.status) {
            result = result.filter(a => a.status === filter.status);
        }
        if (filter.type) {
            result = result.filter(a => a.type === filter.type);
        }
        if (filter.level) {
            result = result.filter(a => a.level === filter.level);
        }

        return result.sort((a, b) => b.timestamp - a.timestamp);
    }

    focusAlarm(alarmId) {
        const alarm = this.alarms.find(a => a.id === alarmId);
        if (alarm && this.viewer) {
            this.viewer.focusOnAlarm(alarm);
        }
        return alarm;
    }

    confirmAlarm(alarmId) {
        const alarm = this.alarms.find(a => a.id === alarmId);
        if (alarm) {
            alarm.status = 'confirmed';
            alarm.confirmedAt = Date.now();
            if (this.handlers.confirm) {
                this.handlers.confirm(alarm);
            }
        }
        return alarm;
    }

    disperseAlarm(alarmId) {
        const alarm = this.alarms.find(a => a.id === alarmId);
        if (alarm) {
            alarm.status = 'dispersing';
            if (this.handlers.disperse) {
                this.handlers.disperse(alarm);
            }
        }
        return alarm;
    }

    reportAlarm(alarmId) {
        const alarm = this.alarms.find(a => a.id === alarmId);
        if (alarm) {
            alarm.status = 'reported';
            alarm.reportedAt = Date.now();
            if (this.handlers.report) {
                this.handlers.report(alarm);
            }
        }
        return alarm;
    }

    resolveAlarm(alarmId) {
        const alarm = this.alarms.find(a => a.id === alarmId);
        if (alarm) {
            alarm.status = 'resolved';
            alarm.resolvedAt = Date.now();
        }
        return alarm;
    }

    onConfirm(handler) {
        this.handlers.confirm = handler;
    }

    onDisperse(handler) {
        this.handlers.disperse = handler;
    }

    onReport(handler) {
        this.handlers.report = handler;
    }
}

// =============================================================================
// 导出
// =============================================================================
window.Indoor3DViewer = Indoor3DViewer;
window.AlarmManager = AlarmManager;

console.log('[Indoor3D] 模块已加载');
