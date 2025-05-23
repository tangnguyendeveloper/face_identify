#ifndef _people_hpp_
#define _people_hpp_

#include <string>
#include <sstream>

class People {
private:
    int id;
    std::string name;
    int old;

public:
    People() : id(0), name(""), old(0) {} // Constructor mặc định
    People(int id, const std::string& name, int old)
        : id(id), name(name), old(old) {}

    People(const People& other) // Copy constructor
        : id(other.id), name(other.name), old(other.old) {}
    
    People& operator=(const People& other) { // Copy assignment operator
        if (this != &other) {
            id = other.id;
            name = other.name;
            old = other.old;
        }
        return *this;
    }

    People(People&& other) noexcept // Move constructor
        : id(other.id), name(std::move(other.name)), old(other.old) {
        other.id = 0;
        other.old = 0;
    }

    People& operator=(People&& other) noexcept { // Move assignment operator
        if (this != &other) {
            id = other.id;
            name = std::move(other.name);
            old = other.old;
            other.id = 0;
            other.old = 0;
        }
        return *this;
    }

    // Getters
    int getId() const { return id; }
    std::string getName() const { return name; }
    int getOld() const { return old; }

    // Setters
    void setId(int newId) { id = newId; }
    void setName(const std::string& newName) { name = newName; }
    void setOld(int newOld) { old = newOld; }

    // equality operator
    bool operator==(const People& other) const {
        return id == other.id && name == other.name && old == other.old;
    }

    // Convert to JSON string
    std::string toJsonString() const {
        std::ostringstream oss;
        oss << "{"
            << "\"id\":" << id << ","
            << "\"name\":\"" << name << "\","
            << "\"old\":" << old
            << "}";
        return oss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const People& p) {
        // rplace space in name with "_"
        std::string name = p.name;
        std::replace(name.begin(), name.end(), ' ', '_');
        os << p.id << " " << name << " " << p.old;
        return os;
    }

    friend std::istream& operator>>(std::istream& is, People& p) {
        is >> p.id >> p.name >> p.old;
        // replace "_" with " " in name
        std::string name = p.name;
        std::replace(name.begin(), name.end(), '_', ' ');
        p.name = name;
        return is;
    }
};

#endif // _people_hpp_