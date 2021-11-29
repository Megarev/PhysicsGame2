#pragma once
#include <map>
#include <set>

#ifdef PHYSICS_LIB_H
#undef PHYSICS_LIB_H
constexpr float PI = 3.1415926f;
const olc::vf2d GRAVITY = { 0.0f, 10.0f };
constexpr float EPSILON = 0.001f;
#endif

// O-----------Math utilities----------O
struct Mat2x2 {
	float m00 = 1.0f, m01 = 0.0f;
	float m10 = 0.0f, m11 = 1.0f;

	Mat2x2() = default;
	Mat2x2(float _m00, float _m01, float _m10, float _m11)
		: m00(_m00), m01(_m01), m10(_m10), m11(_m11) {}

	Mat2x2(float rotation) {
		SetRotation(cosf(rotation), sinf(rotation));
	}

	void SetRotation(float c, float s) {
		m00 = c; m01 = -s;
		m10 = s; m11 = c;
	}

	static Mat2x2 Transpose(Mat2x2 m) {
		return Mat2x2{
			m.m00, m.m10,
			m.m01, m.m11
		};
	}

	olc::vf2d Rotate(const olc::vf2d& input) {
		return {
			input.x * m00 + input.y * m01,
			input.x * m10 + input.y * m11
		};
	}

	friend olc::vf2d operator*(const olc::vf2d& input, const Mat2x2& m) {
		return {
			input.x * m.m00 + input.y * m.m10,
			input.x * m.m01 + input.y * m.m11
		};
	}

	Mat2x2 operator+(const Mat2x2& input) {
		return {
			this->m00 + input.m00, this->m01 + input.m01,
			this->m10 + input.m10, this->m11 + input.m11
		};
	}

	Mat2x2 operator+(float other) {
		return {
			this->m00 + other, this->m01 + other,
			this->m10 + other, this->m11 + other
		};
	}

	Mat2x2 operator-(const Mat2x2& input) {
		return {
			this->m00 - input.m00, this->m01 - input.m01,
			this->m10 - input.m10, this->m11 - input.m11
		};
	}

	Mat2x2 operator*(const Mat2x2& input) {
		return {
			this->m00 * input.m00 + this->m01 * input.m10,
			this->m00 * input.m01 + this->m01 * input.m11,
			this->m10 * input.m00 + this->m11 * input.m10,
			this->m10 * input.m01 + this->m11 * input.m11
		};
	}

	Mat2x2 Invert() {
		float d = (this->m00 * this->m11 - this->m01 * this->m10);
		if (d) {
			float inv_d = 1.0f / d;
			return Mat2x2{ inv_d * this->m11, -inv_d * this->m01, -inv_d * this->m10, inv_d * this->m00 };
		}
		return Mat2x2{};
	}
};

class Util {
public:
	static olc::vf2d Lerp(const olc::vf2d& a, const olc::vf2d& b, float t) {
		return b * t + (1.0f - t) * a;
	}

	static olc::vf2d Cross(const olc::vf2d& a, float s) {
		return { a.y * s, -a.x * s };
	}

	static int SignBool(bool b) {
		return b ? 1 : -1;
	}

	static int Sign(float x) {
		return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
	}

	static bool Contains(float a, float b, float x) {
		if (a > b) std::swap(a, b);
		return a < x&& b > x;
	}

	static float Clamp(float a, float b, float x) {
		if (a > b) std::swap(a, b);
		return std::fmaxf(a, std::fminf(b, x));
	}

	static float Square(float x) { return x * x; }

	static int Random(int a, int b) {
		std::random_device rd;
		static std::mt19937 m(rd());
		std::uniform_int_distribution<> dist(a, b);

		return dist(m);
	}

	static int Random(int n) {
		return Random(0, n);
	}

	static olc::Pixel RandomColor() {
		return olc::Pixel(Random(0, 256), Random(0, 256), Random(0, 256));
	}

	static olc::vi2d RandomPoint(int a, int b) {
		return { Random(a, b), Random(a, b) };
	}

	static bool BiasGreaterThan(float a, float b, float relative = 0.95f, float absolute = 0.05f) {
		return a >= b * relative + a * absolute;
	}
};

// O----------String Formatting----------O
struct StringUtil {
	static std::string ToString(const std::string& format, const std::vector<void*>& parameters) {
		std::string output;
		std::istringstream iss(format);

		std::string s;
		uint32_t index = 0u;
		while (std::getline(iss, s, ' ')) {
			if (s == "%i") output += std::to_string(*(int*)parameters[index++]);
			else if (s == "%f") output += std::to_string(*(float*)parameters[index++]);
			else if (s == "%d") output += std::to_string(*(double*)parameters[index++]);
			else if (s == "%b") {
				bool input = *(bool*)parameters[index++];
				output += (input ? "true" : "false");
			}
			else output += s;

			output += " ";
		}

		return output;
	}

	static std::string ToString2(const std::string& format, const std::vector<void*>& parameters) {
		std::string output;

		uint32_t index = 0u;
		for (int i = 0; i < (int)format.size(); i++) {
			if (format[i] == '%') {
				if (format[i + 1] == 'i') output += std::to_string(*(int*)parameters[index++]);
				else if (format[i + 1] == 'f') output += std::to_string(*(float*)parameters[index++]);
				else if (format[i + 1] == 'd') output += std::to_string(*(double*)parameters[index++]);
				else if (format[i + 1] == 'b') output += (*(bool*)parameters[index++]) ? "true" : "false";
				else output += format.substr(i, 2);
				i++;
			}
			else output += format[i];
		}

		return output;
	}
};

// O----------Pan and Zoom-----------O
struct PanZoom {
	olc::vf2d offset;
	float zoom = 1.0f;

	olc::vf2d prev_m_pos;

	olc::vf2d ToScreen(const olc::vf2d& point) { return (point - offset) * zoom; }
	olc::vf2d ToWorld(const olc::vf2d& point) { return point / zoom + offset; }

	void Update(olc::PixelGameEngine* pge) {
		const olc::vf2d& m_pos = (olc::vf2d)pge->GetMousePos();

		if (pge->GetMouse(2).bHeld) { offset += (m_pos - prev_m_pos) / zoom; }

		const olc::vf2d& m_pos_before_zoom = ToWorld(m_pos);
		if (pge->GetMouseWheel()) { zoom *= (Util::Sign((float)pge->GetMouseWheel()) < 0 ? 1.05f : 0.95f); }
		const olc::vf2d& m_pos_after_zoom = ToWorld(m_pos);
		offset += -(m_pos_after_zoom - m_pos_before_zoom);

		prev_m_pos = m_pos;
	}
};

class Shape {
public:
	olc::vf2d position, velocity;
	float sf = 0.0f, df = 0.0f;
	uint32_t id = 0;
	float scale = 0.0f;

	olc::Pixel color;
	float mass = 0.0f, inv_mass = 0.0f;
	float I = 0.0f, inv_I = 0.0f;

	float e = 0.0f;
	float rotation = 0.0f, angular_velocity = 0.0f;
	Mat2x2 m; // Rotation matrix
	std::vector<olc::vf2d> normals; // Untransformed space
	std::pair<olc::vf2d, olc::vf2d> aabb;

	olc::vf2d force;
	olc::vf2d position_impulse;
	float torque = 0.0f;
public:
	Shape() = default;
	Shape(const olc::vf2d& p, float _scale, const olc::Pixel& col, float _mass, float _I, float _e)
		: position(p), scale(_scale), color(col), mass(_mass), I(_I), e(_e) {}

	void SetFriction(float _sf, float _df) { sf = _sf; df = _df; }

	void IntegrateForces(float dt, float g_multiplier) {
		if (inv_mass == 0.0f) return;

		velocity += (force * inv_mass + GRAVITY * g_multiplier) * dt;
		angular_velocity += (torque * inv_I) * dt;

		force = {};
		torque = 0.0f;
	}

	void IntegrateVelocity(float dt) {
		if (inv_mass == 0.0f) return;

		position += velocity * dt + position_impulse;
		rotation += angular_velocity * dt;

		position_impulse *= 0.8f;
	}

	void AddForce(const olc::vf2d& f) { force += f; }
	void ApplyImpulse(const olc::vf2d& impulse, const olc::vf2d& contact) {
		velocity += impulse * inv_mass;
		angular_velocity += -contact.cross(impulse) * inv_I;
	}

	void Move(const olc::vf2d& d_position) {
		position += d_position;
	}
public:
	virtual void Draw(olc::PixelGameEngine* pge, PanZoom& pan_zoom, bool is_fill, bool is_outline) = 0;
	virtual void Update() = 0;

	virtual bool IsPointInBounds(const olc::vf2d& point) = 0;
};

class PolygonShape : public Shape {
public:
	olc::vf2d size;
	std::vector<olc::vf2d> vertices; // Untransformed space
	int n_vertices = 0;
public:
	PolygonShape() = default;
	PolygonShape(const olc::vf2d& p, const olc::vf2d& s, const olc::Pixel& col, float _mass, float _e, int n)
		: Shape(p, s.mag(), col, _mass, _mass* s.mag2() / 6.0f, _e), size(s), n_vertices(n) {
		inv_mass = mass == 0.0f ? 0.0f : 1.0f / mass;
		inv_I = I == 0.0f ? 0.0f : 1.0f / I;

		for (int i = 0; i < n; i++) {
			vertices.push_back({
				s.x * cosf(2.0f * PI / n * i + PI / 4.0f),
				s.y * sinf(2.0f * PI / n * i + PI / 4.0f)
				});
		}

		for (int i = 0; i < n; i++) {
			normals.push_back(
				(vertices[i] + vertices[(i + 1) % n]).norm()
			);
		}
	}

	PolygonShape(float px, float py, float sx, float sy, const olc::Pixel& col, float _mass, float _e, int n)
		: PolygonShape({ px, py }, { sx, sy }, col, _mass, _e, n) {}

	olc::vf2d GetTransformedVertex(int index) const { return position + GetVertex(index) * m; }

	void Draw(olc::PixelGameEngine* pge, PanZoom& pan_zoom, bool is_fill, bool is_outline) override {

		if (is_fill) {

			const olc::vf2d& v0 = pan_zoom.ToWorld(GetTransformedVertex(0));

			for (int i = 0; i < (int)vertices.size() - 2; i++) {
				pge->FillTriangle(v0, pan_zoom.ToWorld(GetTransformedVertex(i + 1)), pan_zoom.ToWorld(GetTransformedVertex(i + 2)), color);
			}

			if (!is_outline) return;
		}

		for (int i = 0; i < (int)vertices.size(); i++) {
			pge->DrawLine(pan_zoom.ToWorld(vertices[i] * m + position),
				pan_zoom.ToWorld(vertices[(i + 1) % vertices.size()] * m + position), is_outline ? olc::WHITE : color);
		}
	}

	void Update() override {
		m.SetRotation(cosf(rotation), sinf(rotation));

		aabb.first = { INFINITY, INFINITY };
		aabb.second = { -INFINITY, -INFINITY };

		for (auto& v : vertices) {
			const olc::vf2d& transformed_v = position + v * m;
			aabb.first = { std::fminf(aabb.first.x, transformed_v.x), std::fminf(aabb.first.y, transformed_v.y) };
			aabb.second = { std::fmaxf(aabb.second.x, transformed_v.x), std::fmaxf(aabb.second.y, transformed_v.y) };
		}
	}

	bool IsPointInBounds(const olc::vf2d& point) override {
		for (int i = 0; i < (int)vertices.size() - 2; i++) {
			const olc::vf2d& ab = GetTransformedVertex(i + 1) - GetTransformedVertex(0);
			const olc::vf2d& bc = GetTransformedVertex(i + 2) - GetTransformedVertex(i + 1);
			const olc::vf2d& ca = GetTransformedVertex(0) - GetTransformedVertex(i + 2);

			const olc::vf2d& pa = point - GetTransformedVertex(0);
			const olc::vf2d& pb = point - GetTransformedVertex(i + 1);
			const olc::vf2d& pc = point - GetTransformedVertex(i + 2);

			if (ab.cross(pa) < 0.0f && bc.cross(pb) < 0.0f && ca.cross(pc) < 0.0f) return true;
		}
		return false;
	}

	olc::vf2d Support(const olc::vf2d& dir, int& index) {
		float max_distance = -INFINITY;
		index = -1;

		for (int i = 0; i < (int)vertices.size(); i++) {
			const olc::vf2d& r_v = position + vertices[i] * m;
			float d = dir.dot(r_v);

			if (d > max_distance) {
				max_distance = d;
				index = i;
			}
		}

		return position + vertices[index] * m;
	}

	const olc::vf2d& GetVertex(int index) const {
		return vertices[GetVertexIndex(index)];
	}

	int GetVertexIndex(int index) const {
		if (index < 0) return (index + (int)vertices.size()) % (int)vertices.size();
		else return index % (int)vertices.size();
	}
};

class Joint {
public:
	olc::vf2d P; // Impulse cache
	float bias_factor = 0.2f, b = 0.0f;
public:
	Joint() = default;
public:
	virtual void ApplyForces() = 0;
	virtual void PreStep(float inv_dt) = 0;
	virtual void Draw(olc::PixelGameEngine* pge, PanZoom& pan_zoom) = 0;
};


class DistanceJoint : public Joint {
public:
	Shape* shape = nullptr;
	olc::vf2d anchor, local_anchor, bias;
	Mat2x2 M;
public:
	DistanceJoint() = default;
	DistanceJoint(Shape* _shape, const olc::vf2d& _anchor)
		: Joint(), shape(_shape), anchor(_anchor) {
		local_anchor = (anchor - shape->position) * Mat2x2::Transpose(shape->m);
	}

	void PreStep(float inv_dt) override {
		const olc::vf2d& r = local_anchor * shape->m;

		Mat2x2 K1;
		K1.m00 = shape->inv_mass; K1.m01 = 0.0f;
		K1.m10 = 0.0f;			  K1.m11 = shape->inv_mass;

		Mat2x2 K2;
		K2.m00 = shape->inv_I * r.y * r.y; K2.m01 = -shape->inv_I * r.x * r.y;
		K2.m10 = -shape->inv_I * r.x * r.y; K2.m11 = shape->inv_I * r.x * r.x;

		Mat2x2 K = K1 + K2 + b;
		M = K.Invert();

		// Accumulate impulses
		shape->ApplyImpulse(P, r);

		const olc::vf2d& dP = anchor - (shape->position + r);
		bias = bias_factor * inv_dt * dP;
	}

	void ApplyForces() override {
		const olc::vf2d& r = local_anchor * shape->m;
		const olc::vf2d& rv = shape->velocity + Util::Cross(r, shape->angular_velocity);

		const olc::vf2d& impulse = (-rv + bias - b * P) * M;

		shape->ApplyImpulse(impulse, r);

		P += impulse;

		/*const olc::vf2d& r = local_anchor * shape->m;
		const olc::vf2d& dP = anchor - (shape->position + r);

		const olc::vf2d& impulse = k * dP * shape->inv_mass - b * shape->velocity;
		shape->ApplyImpulse(impulse, r);

		P += impulse;*/
	}

	void Draw(olc::PixelGameEngine* pge, PanZoom& pan_zoom) override {
		pge->DrawLine(pan_zoom.ToWorld(anchor), pan_zoom.ToWorld(shape->position), olc::CYAN);
		//DrawUtil::DrawThickLine(pge, pan_zoom.ToWorld(anchor), pan_zoom.ToWorld(shape->position), 5, olc::CYAN, true);
	}
};

class RevoluteJoint : public Joint {
public:
	Shape* shapeA = nullptr, * shapeB = nullptr;
	olc::vf2d local_anchorA, local_anchorB;
	Mat2x2 M;

	olc::vf2d bias;
public:
	RevoluteJoint() = default;
	RevoluteJoint(Shape* _shapeA, Shape* _shapeB)
		: Joint(), shapeA(_shapeA), shapeB(_shapeB) {
		const olc::vf2d& center = (shapeA->position + shapeB->position) / 2.0f;
		local_anchorA = (center - shapeA->position) * Mat2x2::Transpose(shapeA->m);
		local_anchorB = (center - shapeB->position) * Mat2x2::Transpose(shapeA->m);
	}

	void PreStep(float inv_dt) override {
		const olc::vf2d& rA = local_anchorA * shapeA->m;
		const olc::vf2d& rB = local_anchorB * shapeB->m;

		Mat2x2 K1;
		K1.m00 = shapeA->inv_mass + shapeB->inv_mass; K1.m01 = 0.0f;
		K1.m10 = 0.0f;								  K1.m11 = shapeA->inv_mass + shapeB->inv_mass;

		Mat2x2 K2;
		K2.m00 = shapeA->inv_I * rA.y * rA.y;		  K2.m01 = -shapeA->inv_I * rA.y * rA.x;
		K2.m10 = -shapeA->inv_I * rA.x * rA.y;		  K2.m11 = shapeA->inv_I * rA.x * rA.x;

		Mat2x2 K3;
		K3.m00 = shapeB->inv_I * rB.y * rB.y;		  K3.m01 = -shapeB->inv_I * rB.y * rB.x;
		K3.m10 = -shapeB->inv_I * rB.x * rB.y;		  K3.m11 = shapeB->inv_I * rB.x * rB.x;

		Mat2x2 K = K1 + K2 + K3 + b;
		M = K.Invert();

		const olc::vf2d& dP = shapeB->position + rB - (shapeA->position + rA);
		bias = -bias_factor * inv_dt * dP;

		// Accumulate impulses
		shapeA->ApplyImpulse(-P, rA);
		shapeB->ApplyImpulse(P, rB);
	}

	void ApplyForces() override {
		const olc::vf2d& rA = local_anchorA * shapeA->m;
		const olc::vf2d& rB = local_anchorB * shapeB->m;

		const olc::vf2d& rv = shapeB->velocity + Util::Cross(rB, shapeB->angular_velocity) -
			(shapeA->velocity + Util::Cross(rA, shapeA->angular_velocity));

		const olc::vf2d& impulse = (-rv + bias - b * P) * M;

		shapeA->ApplyImpulse(-impulse, rA);
		shapeB->ApplyImpulse(impulse, rB);

		P += impulse;
	}

	void Draw(olc::PixelGameEngine* pge, PanZoom& pan_zoom) override {
		pge->DrawLine(pan_zoom.ToWorld(shapeA->position), pan_zoom.ToWorld(shapeB->position), olc::YELLOW);
	}
};

// O------------Broad phase------------O
class BroadPhase {
public:
	static bool AABBTest(Shape* b1, Shape* b2) {
		return (
			b1->aabb.second.x > b2->aabb.first.x && b2->aabb.second.x > b1->aabb.first.x &&
			b1->aabb.second.y > b2->aabb.first.y && b2->aabb.second.y > b1->aabb.first.y
			);
	}
};

// O------------Narrow phase------------O
class NarrowPhase {
public:
	/*static bool AABB(Shape* b1, Shape* b2, float& overlap, olc::vf2d& normal) {

		float extent = 0.1f;
		const olc::vf2d& b1_size = b1.size + extent * olc::vf2d{ 1.0f, 1.0f };
		const olc::vf2d& b2_size = b2.size + extent * olc::vf2d{ 1.0f, 1.0f };

		const olc::vf2d& distance = { std::fabsf(b2.position.x - b1.position.x), std::fabsf(b2.position.y - b1.position.y) };
		const olc::vf2d& intersection = (b1_size + b2_size) / 2.0f - distance;

		if (!(intersection.x > 0.0f && intersection.y > 0.0f)) return false;

		olc::vf2d overlap_axis_dir = intersection.x < intersection.y ?
			olc::vf2d{ intersection.x, 0.0f } : olc::vf2d{ 0.0f, intersection.y };

		if (overlap_axis_dir.dot(b2.position - b1.position) < 0.0f) overlap_axis_dir *= -1.0f;

		float len = std::fabsf(overlap_axis_dir.dot({ 1.0f, 1.0f }));
		normal = overlap_axis_dir / len;
		overlap = len;

		return true;
	}*/

	static bool PolyonSAT(PolygonShape* b1, PolygonShape* b2, float& overlap, olc::vf2d& normal) {
		auto ProjectOnAxis = [](PolygonShape* b, const olc::vf2d& axis, float& min, float& max) -> void {
			for (auto& v : b->vertices) {

				const olc::vf2d& r_v = b->position + v * b->m;

				min = std::fminf(r_v.dot(axis), min);
				max = std::fmaxf(r_v.dot(axis), max);
			}
		};

		PolygonShape* a = b1, * b = b2;

		overlap = +INFINITY;

		for (int k = 0; k < 2; k++) {
			if (k) std::swap(a, b);

			for (auto& n : a->normals) {

				const olc::vf2d& model_b_normal = (n * a->m);

				float minA = INFINITY, maxA = -INFINITY, minB = INFINITY, maxB = -INFINITY;
				ProjectOnAxis(a, model_b_normal, minA, maxA);
				ProjectOnAxis(b, model_b_normal, minB, maxB);

				if (minA > maxB || minB > maxA) return false;

				float new_overlap = std::fminf(maxA, maxB) - std::fmaxf(minA, minB);

				if (new_overlap < overlap) {
					overlap = new_overlap;
					normal = model_b_normal;
				}
			}
		}

		if (overlap < 0.0f) return false;

		if ((b2->position - b1->position).dot(normal) < 0.0f) normal *= -1.0f;

		return true;
	}
};

// O-----------Contact data-----------O
union FeaturePair {
	struct {
		char inEdge1, outEdge1;
		char inEdge2, outEdge2;
	} e;
	int value = 0;
};

struct ClipVertex {
	olc::vf2d v;
	FeaturePair fp;
};

struct Edge {
	//olc::vf2d a, b, edge, far_vertex;
	ClipVertex a, b;
	olc::vf2d edge, far_vertex;
};

struct Contact {
	olc::vf2d p;
	float overlap;
	olc::vf2d normal;
	FeaturePair fp;
	float bias = 0.0f;

	float Pn = 0.0f, Pt = 0.0f;
	float mn = 0.0f, mt = 0.0f;
};

// O-----------Contact Manifold------------O
class Manifold {
public:
	Shape* a = nullptr, * b = nullptr;
	float total_overlap = 0.0f;
	olc::vf2d normal;
	std::vector<Contact> cp;

	float slop = 0.01f;

private:
	int PolygonPolygonSolve() {
		// SAT
		float overlap = +INFINITY;
		olc::vf2d normal;

		auto ProjectOnAxis = [&](PolygonShape* box, const olc::vf2d& axis, float& min, float& max) {
			for (int i = 0; i < (int)box->vertices.size(); i++) {

				const olc::vf2d& r_v = box->GetTransformedVertex(i);

				min = std::fminf(r_v.dot(axis), min);
				max = std::fmaxf(r_v.dot(axis), max);
			}
		};

		auto GetEdgeID = [](int index, int dir) -> char { return static_cast<char>(std::fminf(index, index + dir) + std::abs(dir)); };

		PolygonShape* pA = static_cast<PolygonShape*>(a), * pB = static_cast<PolygonShape*>(b);
		PolygonShape* bA = pA, * bB = pB;
		for (int k = 0; k < 2; k++) {
			if (k) std::swap(bA, bB);

			for (int i = 0; i < (int)bA->normals.size(); i++) {
				const olc::vf2d& axis = bA->normals[i] * bA->m;

				float min_a = INFINITY, max_a = -INFINITY;
				ProjectOnAxis(bA, axis, min_a, max_a);

				float min_b = INFINITY, max_b = -INFINITY;
				ProjectOnAxis(bB, axis, min_b, max_b);

				if (min_a > max_b || min_b > max_a) return 0;

				float new_overlap = std::fminf(max_a, max_b) - std::fmaxf(min_a, min_b);

				if (new_overlap < overlap) {
					overlap = new_overlap;
					normal = axis;
				}
			}
		}

		if ((pB->position - pA->position).dot(normal) < 0.0f) normal *= -1.0f;

		//if (overlap < 0.0f) return 0;

		// Clip points

		auto GetBestEdge = [&](PolygonShape* box, float sign) -> Edge {
			// Calculate support point
			const olc::vf2d& s_normal = sign * normal;

			int sp_index;
			box->Support(s_normal, sp_index);

			const olc::vf2d& sp = box->GetTransformedVertex(sp_index);

			// Calculate most normal edge
			const olc::vf2d& prev = box->GetTransformedVertex(sp_index + 1);
			const olc::vf2d& next = box->GetTransformedVertex(sp_index - 1);

			const olc::vf2d& prev_sp = sp - prev;
			const olc::vf2d& next_sp = sp - next;

			if ((prev_sp.dot(s_normal)) < (next_sp.dot(s_normal))) {
				ClipVertex v0, v1;
				v0.v = prev;
				v0.fp.e.inEdge2 = GetEdgeID(sp_index + 1, -1);
				v0.fp.e.outEdge2 = GetEdgeID(sp_index + 1, 1);

				v1.v = sp;
				v1.fp.e.inEdge2 = GetEdgeID(sp_index, -1);
				v1.fp.e.outEdge2 = GetEdgeID(sp_index, 1);

				return Edge{ v0, v1, prev_sp, sp };
			}
			else {
				ClipVertex v0, v1;
				v0.v = next;
				v0.fp.e.inEdge2 = GetEdgeID(sp_index - 1, -1);
				v0.fp.e.outEdge2 = GetEdgeID(sp_index - 1, 1);

				v1.v = sp;
				v1.fp.e.inEdge2 = GetEdgeID(sp_index, -1);
				v1.fp.e.outEdge2 = GetEdgeID(sp_index, 1);

				return Edge{ v0, v1, next_sp, sp };
			}
		};

		auto Clip = [](ClipVertex* input, std::vector<ClipVertex>& output, const olc::vf2d& side_normal, float offset, char clip_edge) -> int {
			float d0 = input[0].v.dot(side_normal) - offset;
			float d1 = input[1].v.dot(side_normal) - offset;

			int n = 0;
			//if (d0 >= 0.0f) output[n++] = input[0];
			//if (d1 >= 0.0f) output[n++] = input[1];

			if (d0 >= 0.0f) output.push_back(input[0]);
			if (d1 >= 0.0f) output.push_back(input[1]);

			if (d0 * d1 < 0.0f) {
				float t = d0 / (d0 - d1);
				ClipVertex lv;
				lv.v = Util::Lerp(input[0].v, input[1].v, t);
				if (d0 < 0.0f) {
					lv.fp = input[0].fp;
					lv.fp.e.inEdge1 = clip_edge;
					lv.fp.e.inEdge2 = 0;
				}
				else {
					lv.fp = input[1].fp;
					lv.fp.e.outEdge1 = clip_edge;
					lv.fp.e.outEdge2 = 0;
				}

				output.push_back(lv);
			}

			return (int)output.size();
		};

		const Edge& edgeA = GetBestEdge(pA, 1.0f);
		const Edge& edgeB = GetBestEdge(pB, -1.0f);

		// Get most normal edge
		Edge ref, inc;
		Shape* ref_b, * inc_b;
		bool flip;

		if (std::fabsf(edgeA.edge.dot(normal)) < std::fabsf(edgeB.edge.dot(normal))) {
			ref = edgeA; ref_b = a;
			inc = edgeB; inc_b = b;
			flip = false;
		}
		else {
			ref = edgeB; ref_b = b;
			inc = edgeA; inc_b = a;
			flip = true;
		}

		//DrawUtil::DrawThickLine(pge, pan_zoom.ToWorld(ref.a.v), pan_zoom.ToWorld(ref.b.v), 5, olc::RED);
		//DrawUtil::DrawThickLine(pge, pan_zoom.ToWorld(inc.a.v), pan_zoom.ToWorld(inc.b.v), 5, olc::BLUE);

		// Clip points against the voronoi regions
		const olc::vf2d& ref_edge = ref.edge.norm();

		int n_cp = 0;
		char clip_edge = 0;

		// Positive side voronoi region clip
		ClipVertex in_cv0[2] = { inc.a, inc.b };
		std::vector<ClipVertex> out_cv0;
		clip_edge = in_cv0[1].fp.e.outEdge2;
		n_cp = Clip(in_cv0, out_cv0, ref_edge, ref_edge.dot(ref.a.v), clip_edge);
		if (n_cp < 2) return 0;

		// Negative side voronoi region clip
		ClipVertex in_cv1[2] = { out_cv0[0], out_cv0[1] };
		std::vector<ClipVertex> out_cv1;
		clip_edge = in_cv0[0].fp.e.outEdge2;
		n_cp = Clip(in_cv1, out_cv1, -ref_edge, -ref_edge.dot(ref.b.v), clip_edge);
		if (n_cp < 2) return 0;

		/*for (auto& o : out_cv0) {
			pge->FillCircle(pan_zoom.ToWorld(o.v), 5 / pan_zoom.zoom, olc::YELLOW);
		}

		for (auto& o : out_cv1) {
			pge->FillCircle(pan_zoom.ToWorld(o.v), 2 / pan_zoom.zoom, olc::WHITE);
		}*/

		// Face normal voronoi region clip
		ClipVertex output[2];
		olc::vf2d ref_norm = { -ref_edge.y, ref_edge.x };
		if ((ref_b->position - inc_b->position).dot(ref_norm) < 0.0f) ref_norm *= -1.0f;

		float farthest_distance = ref.far_vertex.dot(ref_norm);
		float d0 = out_cv1[0].v.dot(ref_norm), d1 = out_cv1[1].v.dot(ref_norm);
		float d[2];

		int n_p = 0;
		if (d0 > farthest_distance) {
			output[n_p] = out_cv1[0];
			d[n_p] = d0 - farthest_distance;
			n_p++;
		}
		if (d1 > farthest_distance) {
			output[n_p] = out_cv1[1];
			d[n_p] = d1 - farthest_distance;
			n_p++;
		}

		/*for (auto& o : output) {
			pge->FillCircle(pan_zoom.ToWorld(o.v), 5 / pan_zoom.zoom, olc::GREEN);
		}*/

		auto Flip = [](FeaturePair& fp) {
			std::swap(fp.e.inEdge1, fp.e.inEdge2);
			std::swap(fp.e.outEdge1, fp.e.outEdge2);
		};

		// Generate contact manifold
		for (int i = 0; i < n_p; i++) {
			Contact c;
			c.p = output[i].v - d[i] * normal;
			c.overlap = d[i];
			c.normal = normal;
			c.fp = output[i].fp;

			if (flip) {
				Flip(c.fp);
			}

			cp.push_back(c);

			total_overlap += c.overlap;
		}
		if (n_p == 2) {
			total_overlap /= 2.0f;

			if (std::fabsf(cp[0].p.dot({ 1.0f, 0.0f })) > std::fabsf(cp[1].p.dot({ 1.0f, 0.0f }))) std::swap(cp[0], cp[1]);
		}

		return n_p;
	}
public:
	int SolveConstraints() {
		return PolygonPolygonSolve();
	}

	float e = 0.0f, sf = 0.0f, df = 0.0f;
public:
	Manifold() = default;
	Manifold(Shape* _a, Shape* _b, float _overlap, const olc::vf2d& _normal)
		: a(_a), b(_b), total_overlap(_overlap), normal(_normal) {
		if (a > b) std::swap(a, b);
		Initialize();
	}
	Manifold(Shape* _a, Shape* _b)
		: a(_a), b(_b) {
		if (a > b) std::swap(a, b);
		Initialize();
	}

	void StaticResolve(float p = 1.0f) {
		if (a->inv_mass + b->inv_mass == 0.0f) return;

		float div = 0.5f;
		if (a->inv_mass == 0.0f || b->inv_mass == 0.0f) div = 1.0f;

		const olc::vf2d& correction = std::fmaxf(total_overlap - slop, 0.0f) * p * cp[0].normal * div;
		if (a->inv_mass > 0.0f) a->Move(-correction);
		if (b->inv_mass > 0.0f) b->Move(+correction);
	}

	void UpdateBoxes() {
		// For stacking (To fix floating point errors)
		if (std::fabsf(a->velocity.x) < EPSILON) a->velocity.x = 0.0f;
		if (std::fabsf(b->velocity.x) < EPSILON) b->velocity.x = 0.0f;

		if (cp.size() == 2u) {
			if (std::fabsf(a->angular_velocity) < EPSILON) a->angular_velocity = 0.0f;
			if (std::fabsf(b->angular_velocity) < EPSILON) b->angular_velocity = 0.0f;
		}
	}

	void UpdateManifold(std::vector<Contact>& new_contacts) {

		Contact mergedContacts[2];

		for (int i = 0; i < (int)new_contacts.size(); ++i)
		{
			Contact* cNew = &new_contacts[i];
			int k = -1;
			for (int j = 0; j < (int)cp.size(); ++j)
			{
				Contact* cOld = &cp[j];
				if (cNew->fp.value == cOld->fp.value)
				{
					k = j;
					break;
				}
			}

			mergedContacts[i] = new_contacts[i];
			if (k > -1)
			{
				Contact* c = &mergedContacts[i];
				Contact* cOld = &cp[k];
				//*c = *cNew;
				c->Pn = cOld->Pn;
				c->Pt = cOld->Pt;
			}
		}
		cp.resize(new_contacts.size());
		for (int i = 0; i < new_contacts.size(); ++i) {
			if (i > (int)cp.size() - 1) cp.push_back(mergedContacts[i]);
			else cp[i] = mergedContacts[i];
		}
	}

	void Initialize() {
		e = std::fminf(a->e, b->e);
		sf = std::sqrtf(a->sf * b->sf);
		df = std::sqrtf(a->df * b->df);
	}

	void PreStep(float inv_dt) {
		for (auto& c : cp) {
			const olc::vf2d& ra = c.p - a->position;
			const olc::vf2d& rb = c.p - b->position;

			const olc::vf2d& rv = b->velocity + Util::Cross(rb, b->angular_velocity) -
				(a->velocity + Util::Cross(ra, a->angular_velocity));

			float ra_n = ra.dot(c.normal);
			float rb_n = rb.dot(c.normal);
			c.mn = 1.0f / (a->inv_mass + (ra.mag2() - Util::Square(ra_n)) * a->inv_I + b->inv_mass + (rb.mag2() - Util::Square(rb_n)) * b->inv_I);
			if (std::isnan(c.mn)) c.mn = 0.0f;

			const olc::vf2d& tangent = Util::Cross(c.normal, 1.0f);
			float ra_t = ra.dot(tangent);
			float rb_t = rb.dot(tangent);
			c.mt = 1.0f / (a->inv_mass + (ra.mag2() - Util::Square(ra_t)) * a->inv_I + b->inv_mass + (rb.mag2() - Util::Square(rb_t)) * b->inv_I);
			if (std::isnan(c.mt)) c.mt = 0.0f;

			c.bias = 0.2f * inv_dt * std::fmaxf(0.0f, c.overlap - slop);

			const olc::vf2d& impulse = c.Pn * c.normal + c.Pt * Util::Cross(c.normal, 1.0f);

			a->ApplyImpulse(-impulse, ra);
			b->ApplyImpulse(+impulse, rb);
		}
	}

	void DynamicResolve() {
		for (auto& c : cp) {
			const olc::vf2d& ra = c.p - a->position;
			const olc::vf2d& rb = c.p - b->position;

			olc::vf2d rv = b->velocity + Util::Cross(rb, b->angular_velocity) -
				(a->velocity + Util::Cross(ra, a->angular_velocity));

			float rv_n = (-rv.dot(c.normal) + c.bias) * c.mn;

			float Pn0 = c.Pn;
			c.Pn = std::fmaxf(Pn0 + rv_n, 0.0f);
			float dPn = c.Pn - Pn0;

			const olc::vf2d& impulse_n = dPn * c.normal;
			a->ApplyImpulse(-impulse_n, ra);
			b->ApplyImpulse(+impulse_n, rb);

			rv = b->velocity + Util::Cross(rb, b->angular_velocity) -
				(a->velocity + Util::Cross(ra, a->angular_velocity));

			const olc::vf2d& tangent = Util::Cross(c.normal, 1.0f);

			float jt = -rv.dot(tangent) * c.mt;
			//if (std::fabsf(jt) < EPSILON) continue;

			float Pt = sf * c.Pn;

			float Pt0 = c.Pt;
			c.Pt = Util::Clamp(-Pt, Pt, jt + Pt0);
			float dPt = c.Pt - Pt0;

			const olc::vf2d& impulse_t = dPt * tangent;

			a->ApplyImpulse(-impulse_t, ra);
			b->ApplyImpulse(+impulse_t, rb);
		}
	}

	void Solve() {
		DynamicResolve();
	}

	friend bool operator<(const Manifold& m1, const Manifold& m2) {
		if (m1.a < m2.a) return true;

		if (m1.a == m2.a && m1.b < m2.b) return true;

		return false;
	}
};

struct ManifoldKey {
	Shape* a, * b;

	ManifoldKey(Shape* _a, Shape* _b) {
		if (_a > _b) {
			a = _b;
			b = _a;
		}
		else {
			a = _a;
			b = _b;
		}
	}

	friend bool operator<(const ManifoldKey& m1, const ManifoldKey& m2) {
		if (m1.a < m2.a) return true;

		if (m1.a == m2.a && m1.b < m2.b) return true;

		return false;
	}
};

// O------------Quad Tree------------O
class Boundary {
public:
	olc::vf2d position, size;
public:
	Boundary() = default;
	Boundary(const olc::vf2d& _position, const olc::vf2d& _size)
		: position(_position), size(_size) {}

	bool Contains(const olc::vf2d& point) const noexcept {
		return (point.x >= position.x && point.x <= position.x + size.x &&
			point.y >= position.y && point.y <= position.y + size.y);
	}

	bool Intersects(const Boundary& box) const noexcept {
		return (position.x + size.x > box.position.x && box.position.x + box.size.x > position.x &&
			position.y + size.y > box.position.y && box.position.y + box.size.y > position.y);
	}
};

class QuadTree {
public:
	Boundary box;
	uint8_t capacity;

	std::unique_ptr<QuadTree> branches[4];
	//QuadTree* branches[4]{ nullptr };
	std::vector<Shape*> shapes;
	bool is_divided = false;
private:
	enum { TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT };

	void SubDivideTree() {
		branches[TOP_LEFT] = std::make_unique<QuadTree>(box.position.x, box.position.y, box.size.x / 2.0f, box.size.y / 2.0f, capacity);
		branches[TOP_RIGHT] = std::make_unique<QuadTree>(box.position.x + box.size.x / 2.0f, box.position.y, box.size.x / 2.0f, box.size.y / 2.0f, capacity);
		branches[BOTTOM_RIGHT] = std::make_unique<QuadTree>(box.position.x + box.size.x / 2.0f, box.position.y + box.size.y / 2.0f, box.size.x / 2.0f, box.size.y / 2.0f, capacity);
		branches[BOTTOM_LEFT] = std::make_unique<QuadTree>(box.position.x, box.position.y + box.size.y / 2.0f, box.size.x / 2.0f, box.size.y / 2.0f, capacity);

		//branches[TOP_LEFT] = new QuadTree(box.position.x, box.position.y, box.size.x / 2.0f, box.size.y / 2.0f, capacity);
		//branches[TOP_RIGHT] = new QuadTree(box.position.x + box.size.x / 2.0f, box.position.y, box.size.x / 2.0f, box.size.y / 2.0f, capacity);
		//branches[BOTTOM_RIGHT] = new QuadTree(box.position.x + box.size.x / 2.0f, box.position.y + box.size.y / 2.0f, box.size.x / 2.0f, box.size.y / 2.0f, capacity);
		//branches[BOTTOM_LEFT] = new QuadTree(box.position.x, box.position.y + box.size.y / 2.0f, box.size.x / 2.0f, box.size.y / 2.0f, capacity);

		is_divided = true;
	}
public:
	QuadTree() = default;
	QuadTree(const Boundary& _box, uint8_t _capacity)
		: box(_box), capacity(_capacity) {}
	QuadTree(float px, float py, float sx, float sy, uint8_t _capacity)
		: QuadTree({ { px, py }, { sx, sy } }, _capacity) {}

	void Insert(Shape* b) {

		const Boundary& aabb = { b->aabb.first, (b->aabb.second - b->aabb.first) };
		if (!box.Intersects(aabb)) return;

		if (shapes.size() < capacity) { shapes.push_back(b); }
		else {
			if (!is_divided) SubDivideTree();

			if (branches[TOP_LEFT]) branches[TOP_LEFT]->Insert(b);
			if (branches[TOP_RIGHT]) branches[TOP_RIGHT]->Insert(b);
			if (branches[BOTTOM_RIGHT]) branches[BOTTOM_RIGHT]->Insert(b);
			if (branches[BOTTOM_LEFT]) branches[BOTTOM_LEFT]->Insert(b);
		}
	}

	void DrawTree(olc::PixelGameEngine* pge, const olc::Pixel& color = olc::WHITE) const {
		pge->DrawRect(box.position, box.size - olc::vi2d{ 1, 1 }, color);

		if (is_divided) {
			branches[TOP_LEFT]->DrawTree(pge, color);
			branches[TOP_RIGHT]->DrawTree(pge, color);
			branches[BOTTOM_RIGHT]->DrawTree(pge, color);
			branches[BOTTOM_LEFT]->DrawTree(pge, color);
		}
	}

	void QueryRegion(const Boundary& query_box, std::vector<Shape*>& query_boxes) {
		if (!box.Intersects(query_box)) return;

		for (Shape* const b : shapes) {
			const Boundary& aabb = { b->aabb.first, (b->aabb.second - b->aabb.first) };
			if (query_box.Intersects(aabb)) {
				query_boxes.push_back(b);
			}
		}

		if (is_divided) {
			branches[TOP_LEFT]->QueryRegion(query_box, query_boxes);
			branches[TOP_RIGHT]->QueryRegion(query_box, query_boxes);
			branches[BOTTOM_RIGHT]->QueryRegion(query_box, query_boxes);
			branches[BOTTOM_LEFT]->QueryRegion(query_box, query_boxes);
		}
	}

	void QueryRaycast(const olc::vf2d& ray_a, const olc::vf2d& ray_b, std::vector<std::pair<Shape*, float>>& query_boxes) {
		const olc::vf2d& dir = ray_b - ray_a;

		for (Shape* const b : shapes) {

			olc::vf2d t_near = (b->aabb.first - ray_a) / dir;
			olc::vf2d t_far = (b->aabb.second - ray_a) / dir;

			if (t_near.x > t_far.x) std::swap(t_near.x, t_far.x);
			if (t_near.y > t_far.y) std::swap(t_near.y, t_far.y);

			if (t_near.x > t_far.y || t_near.y > t_far.x) continue;

			float t = std::fmaxf(t_near.x, t_near.y);

			if (t > 0.0f && std::fminf(t_far.x, t_far.y) < 1.0f) query_boxes.push_back({ b, t });
		}

		if (is_divided) {
			branches[TOP_LEFT]->QueryRaycast(ray_a, ray_b, query_boxes);
			branches[TOP_RIGHT]->QueryRaycast(ray_a, ray_b, query_boxes);
			branches[BOTTOM_RIGHT]->QueryRaycast(ray_a, ray_b, query_boxes);
			branches[BOTTOM_LEFT]->QueryRaycast(ray_a, ray_b, query_boxes);
		}
	}

	void QueryRaycastBest(const olc::vf2d& ray_a, const olc::vf2d& ray_b, Shape* best_box, float& t_best) {
		const olc::vf2d& dir = ray_b - ray_a;

		for (Shape* const b : shapes) {

			olc::vf2d t_near = (b->aabb.first - ray_a) / dir;
			olc::vf2d t_far = (b->aabb.second - ray_a) / dir;

			if (std::isnan(t_near.x) || std::isnan(t_near.y) ||
				std::isnan(t_far.x) || std::isnan(t_far.y)) continue;

			if (t_near.x > t_far.x) std::swap(t_near.x, t_far.x);
			if (t_near.y > t_far.y) std::swap(t_near.y, t_far.y);

			if (t_near.x > t_far.y || t_near.y > t_far.x) continue;

			float t1 = std::fmaxf(t_near.x, t_near.y);
			float t2 = std::fminf(t_far.x, t_far.y);

			if (t2 < 0.0f) continue;

			if (t1 > 0.0f && t1 < 1.0f) {
				best_box = b;
				t_best = std::fminf(t_best, t1);
			}
		}

		if (is_divided) {
			branches[TOP_LEFT]->QueryRaycastBest(ray_a, ray_b, best_box, t_best);
			branches[TOP_RIGHT]->QueryRaycastBest(ray_a, ray_b, best_box, t_best);
			branches[BOTTOM_RIGHT]->QueryRaycastBest(ray_a, ray_b, best_box, t_best);
			branches[BOTTOM_LEFT]->QueryRaycastBest(ray_a, ray_b, best_box, t_best);
		}
	}
};

class Scene {
public:
	std::vector<Shape*> shapes;
	std::vector<Joint*> joints;
	olc::vi2d scene_size;

	std::map<ManifoldKey, Manifold> manifold_map;
	uint32_t shape_count = 0u, joint_count = 0u;
	int n_boundary = 2;

	float g_multiplier = 15.0f;

	// For physics step
	float inv_FPS = 1.0f / 60.0f;
	float FPS = 1.0f / inv_FPS;
	float accumulator = 0.0f, delay = 0.1f;

	bool is_boundary_check = false;
public:
	uint32_t AddPolygon(const olc::vf2d& pos, const olc::vf2d& size, const olc::Pixel& color, float mass, int n_vertices, float e = 1.0f, float sf = 0.5f, float df = 0.25f, float rotation = 0.0f) {
		shapes.push_back(new PolygonShape{ pos, size, color, mass, e, n_vertices });
		Shape* b = shapes.back();
		b->rotation = rotation;
		b->SetFriction(sf, df);
		b->id = shape_count++;

		return b->id;
	}

	uint32_t AddPolygon(float px, float py, float sx, float sy, const olc::Pixel& color, float mass, int n_vertices, float e = 1.0f, float sf = 0.2f, float df = 0.25f, float rotation = 0.0f) {
		return AddPolygon({ px, py }, { sx, sy }, color, mass, n_vertices, e, sf, df, rotation);
	}

	uint32_t AddBox(const olc::vf2d& pos, const olc::vf2d& size, const olc::Pixel& color, float mass, float e = 1.0f, float sf = 0.5f, float df = 0.25f, float rotation = 0.0f) {
		return AddPolygon(pos, size, color, mass, 4, e, sf, df, rotation);
	}

	uint32_t AddBox(float px, float py, float sx, float sy, const olc::Pixel& color, float mass, float e = 1.0f, float sf = 0.2f, float df = 0.25f, float rotation = 0.0f) {
		return AddPolygon({ px, py }, { sx, sy }, color, mass, 4, e, sf, df, rotation);
	}

	uint32_t AddDistanceJoint(const olc::vf2d& anchor, Shape* shape, float b = 0.0f) {
		joints.push_back(new DistanceJoint{ shape, anchor });
		joints.back()->b = b;
		return joint_count++;
	}

	uint32_t AddDistanceJoint(float anchor_x, float anchor_y, Shape* shape) {
		return AddDistanceJoint({ anchor_x, anchor_y }, shape);
	}

	uint32_t AddRevoluteJoint(Shape* shapeA, Shape* shapeB, float b = 0.0f) {
		joints.push_back(new RevoluteJoint{ shapeA, shapeB });
		joints.back()->b = b;
		return joint_count++;
	}
public:
	Scene() = default;

	void Update(float dt) {
		accumulator = std::fminf(accumulator + dt, delay);

		while (accumulator > inv_FPS) {
			accumulator -= inv_FPS;

			QuadTree qtree{ 0.0f, 0.0f, (float)scene_size.x, (float)scene_size.y, 5u };
			for (auto& s : shapes) qtree.Insert(s);
			std::vector<std::pair<Shape*, Shape*>> broadphase_pairs;

			// Broadphase
			// Query boxes in a space partition
			int n_boundary = 2;
			float bounds_size = (float)scene_size.x / n_boundary;

			for (int i = 0; i < n_boundary; i++) {
				for (int j = 0; j < n_boundary; j++) {
					Boundary bounds{ { j * bounds_size, i * bounds_size }, { bounds_size, bounds_size } };

					std::vector<Shape*> query_shapes;
					qtree.QueryRegion(bounds, query_shapes);

					for (int a = 0; a < (int)shapes.size() - 1; a++) {
						for (int b = a + 1; b < (int)shapes.size(); b++) {
							if (shapes[a]->inv_mass + shapes[b]->inv_mass == 0.0f) continue;

							ManifoldKey key{ shapes[a], shapes[b] };

							if (BroadPhase::AABBTest(shapes[a], shapes[b])) {
								broadphase_pairs.push_back({ shapes[a], shapes[b] });
							}
							else {
								manifold_map.erase(key);
							}
						}
					}
				}
			}

			std::set<std::pair<Shape*, Shape*>> broadphase_set(broadphase_pairs.begin(), broadphase_pairs.end());
			broadphase_pairs.clear();

			// Narrowphase
			for (auto& pair : broadphase_set) {
				ManifoldKey key{ pair.first, pair.second };
				auto iter = manifold_map.find(key);

				Manifold m{ pair.first, pair.second };
				int np = m.SolveConstraints();

				if (np) {
					if (iter == manifold_map.end()) manifold_map.insert({ key, m });
					else iter->second.UpdateManifold(m.cp);
				}
				else {
					manifold_map.erase(key);
				}
			}

			// Integrate Forces
			for (auto& b : shapes) b->IntegrateForces(inv_FPS, g_multiplier);

			// Solve Presteps
			for (auto& m : manifold_map) m.second.PreStep(1.0f / inv_FPS);
			for (auto& j : joints) j->PreStep(1.0f / inv_FPS);

			// Solve manifolds
			int n_iter = 10;
			for (int i = 0; i < n_iter; i++) {
				for (auto& m : manifold_map) m.second.Solve();
				for (auto& j : joints) j->ApplyForces();
			}

			// Fix errors
			for (auto& m : manifold_map) m.second.UpdateBoxes();

			for (auto& b : shapes) {
				// Integrate velocities
				b->IntegrateVelocity(inv_FPS);

				// Update rotation matrix
				b->Update();
			}

			// Boundary checks
			if (is_boundary_check) {
				for (auto it = shapes.begin(); it != shapes.end();) {
					// Check against side walls
					int left_sp, right_sp;
					PolygonShape* p = static_cast<PolygonShape*>(*it);
					const olc::vf2d& left = p->Support({ -1.0f, 0.0f }, left_sp);
					const olc::vf2d& right = p->Support({ 1.0f, 0.0f, }, right_sp);

					if (left.x < -(float)scene_size.x * 0.5f || right.x > 1.5f * scene_size.y) it = shapes.erase(it);
					else it++;
				}
			}
		}
	}

	void SetBoundaryCheck(bool state) { is_boundary_check = state; }
	void SetSceneSize(const olc::vi2d& size) { scene_size = size; }
};
