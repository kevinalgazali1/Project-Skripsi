<?php

namespace App\Models;

// use Illuminate\Contracts\Auth\MustVerifyEmail;
use App\Models\MitraProfile;
use App\Models\AlumniSiswaProfile;
use App\Models\Sertifikasi;
use App\Models\Loker;
use App\Models\Pelatihan;
use App\Models\DaftarSertifikasi;
use App\Models\DaftarLoker;
use App\Models\DaftarPelatihan;
use Illuminate\Notifications\Notifiable;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Foundation\Auth\User as Authenticatable;

class User extends Authenticatable
{
    /** @use HasFactory<\Database\Factories\UserFactory> */
    use HasFactory, Notifiable;

    /**
     * The attributes that are mass assignable.
     *
     * @var list<string>
     */
    protected $fillable = [
        'email',
        'no_hp',
        'role',
        'password',
    ];

    /**
     * The attributes that should be hidden for serialization.
     *
     * @var list<string>
     */
    protected $hidden = [
        'password',
        'remember_token',
    ];

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'email_verified_at' => 'datetime',
            'password' => 'hashed',
        ];
    }

    public function mitraProfile()
    {
        return $this->hasOne(MitraProfile::class);
    }

    public function alumniSiswaProfile()
    {
        return $this->hasOne(AlumniSIswaProfile::class);
    }

    public function sertifikasi()
    {
        return $this->hasMany(Sertifikasi::class);
    }

    public function loker()
    {
        return $this->hasMany(Loker::class);
    }

    public function pelatihan()
    {
        return $this->hasMany(Pelatihan::class);
    }

    public function daftarSertifikasi()
    {
        return $this->hasMany(DaftarSertifikasi::class);
    }

    public function daftarLoker()
    {
        return $this->hasMany(DaftarLoker::class);
    }

    public function daftarPelatihan ()
    {
        return $this->hasMany(DaftarPelatihan::class);
    }

    public function artikel ()
    {
        return $this->hasMany(Artikel::class);
    }

}
